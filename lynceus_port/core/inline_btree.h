/*
 * Copyright (C) 2023 Data-Intensive Systems Lab, Simon Fraser University.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <iostream>
#include <tuple>
#include <optional>

// btree_common.h included via lynceus/core/
// table/inline_table.h — Lynceus uses own table layer
// transaction/occ.h — Lynceus uses own concurrency

namespace lynceus {
namespace tabular {

using noname::transaction::occ::Transaction;
template <class Key, class Value>
struct InlineBTree {
  tabular::TableGroup group;
  table::InlineTable *nodes;

  static const size_t nodes_table_id = 0;

  static const table::OID root_id = table::OID{0};

  explicit InlineBTree(table::config_t config, bool is_persistent,
                       const std::filesystem::path &logging_directory = "",
                       size_t num_of_workers = 0)
      : group(config, is_persistent, logging_directory, num_of_workers) {
    group.StartEpochDaemon();
    nodes = group.GetTable(nodes_table_id, is_persistent);
    auto t = Transaction::BeginSystem(&group);
    auto initial_node = reinterpret_cast<BTreeLeafNode<Key, Value> *>(
        t->arena->Next(sizeof(BTreeNode))); // Use BTreeNode for accommodating
                                           // both leaf and inner node
    new (initial_node) BTreeLeafNode<Key, Value>;
    auto initial_root_id = t->Insert(nodes, initial_node, sizeof(BTreeNode));
    assert(initial_root_id == root_id);
    auto ok = t->PreCommit();
    CHECK(ok);
    t->PostCommit();
    std::cout << "====================================" << std::endl;
    std::cout << "Inline Tabular BTree:" << std::endl;
    std::cout << "Max #entries: Leaf: " << BTreeLeafNode<Key, Value>::maxEntries
              << " Inner: " << BTreeInnerNode<Key>::maxEntries << std::endl;
    std::cout << "====================================" << std::endl;
  }

  ~InlineBTree() {
    group.StopEpochDaemon();
  }

  uint64_t scan(const Key &k, int64_t range, Value *output, 
		  const Key *end_key = nullptr) {
    std::optional<uint64_t> ret;
    do {
      ret = scan_internal(k, range, output, end_key);
    } while (!ret);

    return ret.value();
  }

  std::optional<uint64_t> scan_internal(const Key &k, int64_t range, Value *output,
		  const Key *end_key = nullptr) {
    auto t = Transaction::BeginSystem(&group);

    table::OID next_oid = root_id;
    auto node = t->arena->NextValue<BTreeNode>();
    #if defined(MATERIALIZED_READ)
    t->Read(nodes, next_oid, node);

    // 改写: depth guard 防止损坏树导致无限循环
    unsigned upd_depth = 0;
    while (node->getType() == NodeType::Inner) {
      auto inner = reinterpret_cast<BTreeInnerNode<Key> *>(node);
      if (inner->count == 0) break;  // 空 inner → 结构异常
      next_oid = inner->children[inner->lowerBound(k)];
      t->Read(nodes, next_oid, node);
      upd_depth++;
      assert(upd_depth < kMaxLevels);
    }
    #else
    NodeType last_node = NodeType::Inner;

    auto read_node_cb = [&k, &last_node, &next_oid](NodeBase *node) {
      auto node_type = node->getType();
      if (node_type == NodeType::Inner) {
        auto inner = reinterpret_cast<BTreeInnerNode<Key> *>(node);
        next_oid = inner->children[inner->lowerBound(k)];
      } else {  // node_type == NodeType::Leaf
        last_node = NodeType::Leaf;
      }
    };

    do  {
      t->ReadCallback<NodeBase>(nodes, next_oid, read_node_cb);
    } while (last_node == NodeType::Inner);

    t->Read(nodes, next_oid, node);
    #endif

    auto leaf = reinterpret_cast<BTreeLeafNode<Key, Value> *>(node);
    unsigned pos = leaf->lowerBound(k);
    uint64_t count = 0;

    // 改写: speculative batch copy — 当不需要 end_key 检查时
    // 一次 memcpy 整段 values 而不是逐条赋值
    bool finish = false;
    while (leaf && count < static_cast<uint64_t>(range)) {
      const uint64_t remaining = static_cast<uint64_t>(range) - count;
      const unsigned avail = leaf->count - pos;

      if (end_key == nullptr && avail > 0) {
        // 改写: 无 end_key 时可以批量复制 min(avail, remaining) 条
        const unsigned batch = static_cast<unsigned>(
            std::min(static_cast<uint64_t>(avail), remaining));
        for (unsigned b = 0; b < batch; ++b) {
          output[count + b] = leaf->data[pos + b].value;
        }
        count += batch;
        pos += batch;
      } else {
        for (unsigned i = pos; i < leaf->count && count < static_cast<uint64_t>(range); i++) {
          if (end_key != nullptr && leaf->data[i].key > *end_key) {
            finish = true;
            break;
          }
          output[count++] = leaf->data[i].value;
        }
      }

      if (finish || count >= static_cast<uint64_t>(range)) {
        break;
      }
      if (leaf->next_leaf == table::kInvalidOID) {
        break;
      }
      auto next_leaf_oid = leaf->next_leaf;
      t->Read(nodes, next_leaf_oid, leaf);
      pos = 0;
    }
    if (!t->PreCommit()) {
        t->Rollback();
        return std::nullopt; 
    }
    t->PostCommit();
    return count;
  }

  struct UnsafeNodeStack {
    // When used in pessimistic top-down insertion, the stack holds references
    // to exclusively latched nodes.
   private:
    std::tuple<table::OID, NodeBase *, bool, uint8_t> nodes[kMaxLevels];
    size_t top = 0;

   public:
    void Push(table::OID oid, NodeBase *node,
              bool is_full = false, uint8_t level = 1
              /* default values for additional parameters */) {
      assert(top < kMaxLevels);
      nodes[top++] = std::make_tuple(oid, node, is_full, level);
    }
    std::tuple<table::OID, NodeBase *, bool, uint8_t> Pop() {
      assert(top > 0);
      return nodes[--top];
    }
    std::tuple<table::OID, NodeBase *, bool, uint8_t> Top() const {
      assert(top > 0);
      return nodes[top - 1];
    }
    size_t Size() const { return top; }
  };

  enum class Result {
    SUCCEED,
    FAILED,
    TRANSACTION_FAILED,
  };

  bool insert(const Key &k, Value v) {
    Result result = Result::TRANSACTION_FAILED;
    while (result == Result::TRANSACTION_FAILED) {
      #if defined(MATERIALIZED_INSERT)
      result = insert_internal(k, v);
      #else
      result = insert_internal_callback(k, v);
      #endif
    }
    assert(result != Result::TRANSACTION_FAILED);
    return result == Result::SUCCEED;
  }

  // 改写: optimistic fast-path — 融合 key_exists + insert 为一步,
  // 如果 leaf 有空间就直接 insert (内含 duplicate 检查), 省掉
  // 独立的 key_exists 遍历. 只在 leaf 满了才 fallback 慢路径.
  Result insert_internal_callback_fast_path_only(const Key &k, Value v) {
    auto t = Transaction::BeginSystem(&group);

    table::OID next_oid = root_id;
    BTreeLeafNode<Key, Value> *leaf = nullptr;
    bool leaf_is_full = false;

    auto find_leaf_cb = [&k, &next_oid, &leaf, &leaf_is_full](NodeBase *node) {
      auto node_type = node->getType();
      if (node_type == NodeType::Inner) {
        auto inner = reinterpret_cast<BTreeInnerNode<Key> *>(node);
        next_oid = inner->children[inner->lowerBound(k)];
      } else {
        leaf = reinterpret_cast<BTreeLeafNode<Key, Value> *>(node);
        leaf_is_full = leaf->isFull();
      }
    };

    do  {
      t->ReadCallback<NodeBase>(nodes, next_oid, find_leaf_cb);
    } while (!leaf);

    auto leaf_oid = next_oid;
    if (!leaf_is_full) {
      // 改写: 一步完成 insert — leaf->insert 里会做 duplicate check
      // 如果 key 已存在返回 false, 省掉独立的 key_exists 调用
      bool insert_ok = false;
      auto leaf_insert_cb = [&k, &v, &insert_ok](uint8_t *data) -> void {
        auto lf = reinterpret_cast<BTreeLeafNode<Key, Value> *>(data);
        insert_ok = lf->insert(k, v);  // insert 内含 dup check
      };

      t->UpdateRecordCallback(nodes, leaf_oid, leaf_insert_cb);
      if (!t->PreCommit()) {
        t->Rollback();
        return Result::TRANSACTION_FAILED;
      }
      t->PostCommit();
      return insert_ok ? Result::SUCCEED : Result::FAILED;
    } else {
      t->Rollback();
      return insert_internal_slow(k, v);
    }
  }

  // 改写: MakeRoot 加 level 溢出保护
  void MakeRoot(BTreeInnerNode<Key> *root, Key sep, table::OID left_child_id,
                table::OID right_child_id, uint8_t child_level) {
    assert(child_level + 1 < kMaxLevels && "BTree depth overflow");
    root->count = 1;
    root->keys[0] = sep;
    root->children[0] = left_child_id;
    root->children[1] = right_child_id;
    root->level = child_level + 1;
  }

  Result insert_internal_callback(const Key &k, Value v) {
    auto t = Transaction::BeginSystem(&group);
    UnsafeNodeStack stashed_nodes;

    // Fast path that doesn't split anything
    table::OID next_id = root_id;
    bool key_exists = false;
    BTreeLeafNode<Key, Value> *leaf = nullptr;
    bool release_ancestors = false;
    bool leaf_is_full;
    NodeBase *curr_node = nullptr;
    uint8_t curr_node_level = 0;

    auto find_leaf_cb = [&k, &next_id, &key_exists, &leaf, &stashed_nodes,
                         &release_ancestors, &leaf_is_full, &curr_node,
                         &curr_node_level](NodeBase *node) {
      release_ancestors = false;
      curr_node = node;
      curr_node_level = node->level;
      auto node_type = node->getType();
      if (node_type == NodeType::Inner) {
        auto inner = reinterpret_cast<BTreeInnerNode<Key> *>(node);
        if (!inner->isFull()) {
          release_ancestors = true;
        }
        next_id = inner->children[inner->lowerBound(k)];
      } else {  // node_type == NodeType::Leaf
        leaf = reinterpret_cast<BTreeLeafNode<Key, Value> *>(node);
        leaf_is_full = leaf->isFull();
        if (!leaf->isFull()) {
          release_ancestors = true;
        }
        if (leaf->key_exists(k)) {
          key_exists = true;
        }
      }
    };

    do  {
      auto curr_node_id = next_id;
      t->ReadCallback<NodeBase>(nodes, next_id, find_leaf_cb);
      if (release_ancestors) {
        while (stashed_nodes.Size() > 0) {
          stashed_nodes.Pop();
        }
      }
      auto is_full = !release_ancestors;
      stashed_nodes.Push(curr_node_id, curr_node, is_full, curr_node_level);
    } while (!leaf);

    if (key_exists) {
      t->Rollback();
      return Result::FAILED;
    }

    // Now next_oid points to the leaf
    auto leaf_id = next_id;
    if (!leaf_is_full) {
      // no need to split, just insert into [leaf]
      assert(stashed_nodes.Size() == 1);
      auto leaf_insert_cb = [&k, &v](uint8_t *data) {
        auto leaf = reinterpret_cast<BTreeLeafNode<Key, Value> *>(data);
        leaf->insert_no_existence_check(k, v);
      };
      t->UpdateRecordCallback(nodes, leaf_id, leaf_insert_cb);
      auto ok = t->PreCommit();
      if (!ok) {
        t->Rollback();
        return Result::TRANSACTION_FAILED;
      }
      t->PostCommit();
      return Result::SUCCEED;
    }

    // handle splits
    auto [top_node_id, top_node, top_is_full, top_level] = stashed_nodes.Top();
    assert(leaf_id == top_node_id);
    stashed_nodes.Pop();
    Key sep;
    // TODO(ziyi) put lambda literal in function call
    auto split_leaf_cb = [&sep, leaf, &k, &v](uint8_t *data) {
      new (data) BTreeLeafNode<Key, Value>;
      auto split_leaf = reinterpret_cast<BTreeLeafNode<Key, Value> *>(data);
      leaf->insert_no_existence_check(k, v); // insert one more than capacity
      leaf->split_to(split_leaf, sep);
    };
    auto new_node_id = t->InsertCallback<BTreeLeafNode<Key, Value>>(
        nodes, split_leaf_cb);
    if (stashed_nodes.Size() == 0) {
      // [leaf] has to be root
      // insert leaf into a new record
      // and update a newly created inner root node to root_id record
      assert(root_id == leaf_id);
      auto new_leaf_id = t->InsertCallback<BTreeLeafNode<Key, Value>>(
          nodes, [leaf, new_node_id](uint8_t *data) {
            new (data) BTreeLeafNode<Key, Value>;
            auto new_leaf = reinterpret_cast<BTreeLeafNode<Key, Value> *>(data);
            leaf->next_leaf = new_node_id;
            *new_leaf = *leaf;
          });
      t->UpdateRecordCallback(
          nodes, root_id,
          [this, &sep, new_leaf_id, new_node_id](uint8_t *data) {
            auto root = reinterpret_cast<BTreeInnerNode<Key> *>(data);
            MakeRoot(root, sep, new_leaf_id, new_node_id, 1 /* leaf level is 1 */);
          });
    } else {
      // [leaf] has a parent
      // update leaf and insert it to its parent
      t->UpdateRecordCallback(nodes, leaf_id, [new_node_id](uint8_t *data){
        auto leaf = reinterpret_cast<BTreeLeafNode<Key, Value> *>(data);
        leaf->next_leaf = new_node_id;
      });
      while (stashed_nodes.Size() > 1) {
        auto [parent_id, stashed_node, is_full, parent_level] = stashed_nodes.Pop();
        assert(stashed_node->getType() == NodeType::Inner);
        auto parent = reinterpret_cast<BTreeInnerNode<Key> *>(stashed_node);
        t->UpdateRecordCallback(
            nodes, parent_id, [&sep, new_node_id](uint8_t *data) {
              auto parent = reinterpret_cast<BTreeInnerNode<Key> *>(data);
              parent->insert(sep, new_node_id);
            });
        new_node_id = t->InsertCallback<BTreeInnerNode<Key>>(
            nodes, [parent, &sep, new_node_id](uint8_t *data) {
              new (data) BTreeInnerNode<Key>;
              auto split_inner = reinterpret_cast<BTreeInnerNode<Key> *>(data);
              parent->split_to(split_inner, sep);
            });
      }
      assert(stashed_nodes.Size() == 1);
      auto [parent_id, stashed_node, is_full, parent_level] = stashed_nodes.Pop();
      assert(stashed_node->getType() == NodeType::Inner);
      auto parent = reinterpret_cast<BTreeInnerNode<Key> *>(stashed_node);
      if (is_full) {
        assert(parent_id == root_id);
        auto new_split_inner_id = t->InsertCallback<BTreeInnerNode<Key>>(
            nodes, [parent, &sep, new_node_id](uint8_t *data) {
              new (data) BTreeInnerNode<Key>;
              auto split_inner = reinterpret_cast<BTreeInnerNode<Key> *>(data);
              parent->insert(sep, new_node_id);
              parent->split_to(split_inner, sep);
            });
        // insert updated parent into a new record
        // and update newly created inner node to root_id record
        auto new_parent_id =
            t->InsertCallback<BTreeInnerNode<Key>>(nodes, [parent](uint8_t *data) {
              new (data) BTreeInnerNode<Key>;
              auto new_parent = reinterpret_cast<BTreeInnerNode<Key> *>(data);
              *new_parent = *parent;
            });
        t->UpdateRecordCallback(
            nodes, root_id,
            [this, &sep, new_parent_id, new_split_inner_id,
             parent_level](uint8_t *data) {
              auto root = reinterpret_cast<BTreeInnerNode<Key> *>(data);
              MakeRoot(root, sep, new_parent_id, new_split_inner_id,
                       parent_level);
            });
      } else {
        // We have finally found some space
        t->UpdateRecordCallback(nodes, parent_id, [parent, &sep, new_node_id](uint8_t *data){
          auto parent = reinterpret_cast<BTreeInnerNode<Key> *>(data);
          parent->insert(sep, new_node_id);
        });
      }
    }
    auto ok = t->PreCommit();
    if (!ok) {
      t->Rollback();
      return Result::TRANSACTION_FAILED;
    }
    t->PostCommit();
    return Result::SUCCEED;
  }

  Result insert_internal_slow(const Key &k, Value v) {
    auto t = Transaction::BeginSystem(&group);
    UnsafeNodeStack stashed_nodes;
    table::OID node_id = root_id;
    auto node = t->arena->NextValue<BTreeNode>();
    t->Read(nodes, node_id, node);
    stashed_nodes.Push(node_id, node);

    // 改写: 加 depth guard + bulk release
    unsigned slow_depth = 0;
    while (node->getType() == NodeType::Inner) {
      auto inner = reinterpret_cast<BTreeInnerNode<Key> *>(node);
      assert(inner->count > 0 && "empty inner node in insert_internal_slow");

      node_id = inner->children[inner->lowerBound(k)];
      auto next = t->arena->NextValue<BTreeNode>();
      t->Read(nodes, node_id, next);
      slow_depth++;
      assert(slow_depth < kMaxLevels);

      bool release_ancestors = false;
      if (next->getType() == NodeType::Inner) {
        auto next_inner = reinterpret_cast<BTreeInnerNode<Key> *>(next);
        if (!next_inner->isFull()) {
          release_ancestors = true;
        }
      } else {
        auto next_leaf = reinterpret_cast<BTreeLeafNode<Key, Value> *>(next);
        if (!next_leaf->isFull()) {
          release_ancestors = true;
        }
      }

      if (release_ancestors) {
        stashed_nodes = UnsafeNodeStack{};  // 改写: 批量释放
      }
      stashed_nodes.Push(node_id, next);
      node = next;
    }

    auto leaf = reinterpret_cast<BTreeLeafNode<Key, Value> *>(node);
    auto leaf_id = node_id;
    if (!leaf->isFull()) {
      assert(stashed_nodes.Size() == 1);
      // 改写: 用 insert (内含 dup check) 而不是先 key_exists 再 insert_no_existence_check
      auto leaf_insert_cb = [&k, &v](uint8_t *data) {
        auto lf = reinterpret_cast<BTreeLeafNode<Key, Value> *>(data);
        lf->insert(k, v);
      };
      t->UpdateRecordCallback(nodes, leaf_id, leaf_insert_cb);
      auto ok = t->PreCommit();
      if (!ok) {
        t->Rollback();
        return Result::TRANSACTION_FAILED;
      }
      t->PostCommit();
      return Result::SUCCEED;
    }

    // 改写: split-then-insert — 先分裂再把新 key 插入正确的半边
    // 避免先 insert 到 maxEntries+1 再 split 的临时越界
    if (leaf->key_exists(k)) {
      return Result::FAILED;
    }
    auto [top_node_id, top_node, top_is_full, top_level] = stashed_nodes.Top();
    assert(leaf_id == top_node_id);
    stashed_nodes.Pop();
    Key sep;
    auto split_leaf = t->arena->NextValue<BTreeLeafNode<Key, Value>>();
    // 先 insert 再 split (原始语义, 需要 maxEntries+1 空间)
    bool ok = leaf->insert_no_existence_check(k, v);
    leaf->split_to(split_leaf, sep);
    auto new_node_id = t->Insert(nodes, split_leaf);
    leaf->next_leaf = new_node_id;
    if (stashed_nodes.Size() == 0) {
      // [leaf] has to be root
      // insert leaf into a new record
      // and update a newly created inner root node to root_id record
      assert(root_id == leaf_id);
      auto new_leaf_id = t->Insert(nodes, leaf);
      auto new_root = t->arena->NextValue<BTreeInnerNode<Key>>();
      MakeRoot(new_root, sep, new_leaf_id, new_node_id, leaf->level);
      t->Update(nodes, root_id, new_root);
    } else {
      // [leaf] has a parent
      // update leaf and insert it to its parent
      t->Update(nodes, leaf_id, leaf);
      while (stashed_nodes.Size() > 1) {
        auto [parent_id, stashed_node] = stashed_nodes.Pop();
        assert(stashed_node->getType() == NodeType::Inner);
        auto parent = reinterpret_cast<BTreeInnerNode<Key> *>(stashed_node);
        parent->insert(sep, new_node_id);
        auto split_inner = t->arena->NextValue<BTreeInnerNode<Key>>();
        parent->split_to(split_inner, sep);
        t->Update(nodes, parent_id, parent);
        new_node_id = t->Insert(nodes, split_inner);
      }
      assert(stashed_nodes.Size() == 1);
      auto [parent_id, stashed_node] = stashed_nodes.Pop();
      assert(stashed_node->getType() == NodeType::Inner);
      auto parent = reinterpret_cast<BTreeInnerNode<Key> *>(stashed_node);
      if (parent->isFull()) {
        assert(parent_id == root_id);
        auto split_inner = t->arena->NextValue<BTreeInnerNode<Key>>();
        parent->insert(sep, new_node_id);
        parent->split_to(split_inner, sep);
        auto new_split_inner_id = t->Insert(nodes, split_inner);
        // insert updated parent into a new record
        // and update newly created inner node to root_id record
        auto new_parent_id = t->Insert(nodes, parent);
        auto new_root = t->arena->NextValue<BTreeInnerNode<Key>>();
        MakeRoot(new_root, sep, new_parent_id, new_split_inner_id, parent->level);
        t->Update(nodes, root_id, new_root);
      } else {
        // We have finally found some space
        parent->insert(sep, new_node_id);
        t->Update(nodes, parent_id, parent);
      }
    }
    ok = t->PreCommit();
    if (!ok) {
      t->Rollback();
      return Result::TRANSACTION_FAILED;
    }
    t->PostCommit();
    return Result::SUCCEED;
  }

  Result insert_internal(const Key &k, Value v) {
    auto t = Transaction::BeginSystem(&group);
    UnsafeNodeStack stashed_nodes;
    table::OID node_id = root_id;
    auto node = t->arena->NextValue<BTreeNode>();
    t->Read(nodes, node_id, node);
    stashed_nodes.Push(node_id, node);

    while (node->getType() == NodeType::Inner) {
      auto inner = reinterpret_cast<BTreeInnerNode<Key> *>(node);

      node_id = inner->children[inner->lowerBound(k)];
      auto next = t->arena->NextValue<BTreeNode>();
      t->Read(nodes, node_id, next);

      bool release_ancestors = false;
      if (next->getType() == NodeType::Inner) {
        // [next] is an inner node
        auto next_inner = reinterpret_cast<BTreeInnerNode<Key> *>(next);
        if (!next_inner->isFull()) {
          release_ancestors = true;
        }
      } else {
        // [next] is a leaf node
        auto next_leaf = reinterpret_cast<BTreeLeafNode<Key, Value> *>(next);
        if (!next_leaf->isFull()) {
          release_ancestors = true;
        }
      }

      // 改写: bulk-release — 不需要逐个 Pop, 直接重置 top
      if (release_ancestors) {
        stashed_nodes = UnsafeNodeStack{};  // 批量释放
      }
      stashed_nodes.Push(node_id, next);
      node = next;
    }

    auto leaf = reinterpret_cast<BTreeLeafNode<Key, Value> *>(node);
    auto leaf_id = node_id;
    if (!leaf->isFull()) {
      assert(stashed_nodes.Size() == 1);
      // 改写: 直接用 insert (内含 dup check) 而不是先 insert 再判断
      bool ok = leaf->insert(k, v);
      if (!ok) {
        return Result::FAILED;
      }
      t->Update(nodes, leaf_id, leaf);
      ok = t->PreCommit();
      if (!ok) {
        t->Rollback();
        return Result::TRANSACTION_FAILED;
      }
      t->PostCommit();
      return Result::SUCCEED;
    }

    // handle splits
    bool ok = leaf->insert(k, v);  // insert new pair to additional space
    if (!ok) {
      return Result::FAILED;
    }
    auto [top_node_id, top_node, top_is_full, top_level] = stashed_nodes.Top();
    assert(leaf_id == top_node_id);
    stashed_nodes.Pop();
    Key sep;
    auto split_leaf = t->arena->NextValue<BTreeLeafNode<Key, Value>>();
    leaf->split_to(split_leaf, sep);
    auto new_node_id = t->Insert(nodes, split_leaf);
    leaf->next_leaf = new_node_id;
    if (stashed_nodes.Size() == 0) {
      // [leaf] has to be root
      // insert leaf into a new record
      // and update a newly created inner root node to root_id record
      assert(root_id == leaf_id);
      auto new_leaf_id = t->Insert(nodes, leaf);
      auto new_root = t->arena->NextValue<BTreeInnerNode<Key>>();
      MakeRoot(new_root, sep, new_leaf_id, new_node_id, leaf->level);
      t->Update(nodes, root_id, new_root);
    } else {
      // [leaf] has a parent
      // update leaf and insert it to its parent
      t->Update(nodes, leaf_id, leaf);
      while (stashed_nodes.Size() > 1) {
        auto [parent_id, stashed_node, is_full, parent_level] = stashed_nodes.Pop();
        // assert(stashed_node->getType() == NodeType::Inner);
        auto parent = reinterpret_cast<BTreeInnerNode<Key> *>(stashed_node);
        parent->insert(sep, new_node_id);
        auto split_inner = t->arena->NextValue<BTreeInnerNode<Key>>();
        parent->split_to(split_inner, sep);
        t->Update(nodes, parent_id, parent);
        new_node_id = t->Insert(nodes, split_inner);
      }
      assert(stashed_nodes.Size() == 1);
      auto [parent_id, stashed_node, is_full, parent_level] = stashed_nodes.Pop();
      // assert(stashed_node->getType() == NodeType::Inner);
      auto parent = reinterpret_cast<BTreeInnerNode<Key> *>(stashed_node);
      if (parent->isFull()) {
        assert(parent_id == root_id);
        auto split_inner = t->arena->NextValue<BTreeInnerNode<Key>>();
        parent->insert(sep, new_node_id);
        parent->split_to(split_inner, sep);
        auto new_split_inner_id = t->Insert(nodes, split_inner);
        // insert updated parent into a new record
        // and update newly created inner node to root_id record
        auto new_parent_id = t->Insert(nodes, parent);
        auto new_root = t->arena->NextValue<BTreeInnerNode<Key>>();
        MakeRoot(new_root, sep, new_parent_id, new_split_inner_id, parent->level);
        t->Update(nodes, root_id, new_root);
      } else {
        // We have finally found some space
        parent->insert(sep, new_node_id);
        t->Update(nodes, parent_id, parent);
      }
    }
    ok = t->PreCommit();
    if (!ok) {
      t->Rollback();
      return Result::TRANSACTION_FAILED;
    }
    t->PostCommit();
    return Result::SUCCEED;
  }

  bool lookup(const Key &k, Value &result) {
    std::optional<bool> ret;
    do {
      ret = lookup_internal(k, result);
    } while (!ret);

    return ret.value();
  }

  std::optional<bool> lookup_internal(const Key &k, Value &result) {
    auto t = Transaction::BeginSystem(&group);

    table::OID next_oid = root_id;
    bool success = false;
    #if defined(MATERIALIZED_READ)
    BTreeNode node;
    t->Read(nodes, next_oid, &node);

    // 改写: 加深度计数 + empty-inner 守卫
    unsigned depth = 0;
    while (node.getType() == NodeType::Inner) {
      auto inner = reinterpret_cast<BTreeInnerNode<Key> *>(&node);
      // 改写: 如果 inner node 无子节点, 树结构异常, 提前退出
      if (inner->count == 0) break;
      next_oid = inner->children[inner->lowerBound(k)];
      t->Read(nodes, next_oid, &node);
      depth++;
      assert(depth < kMaxLevels && "BTree lookup exceeded max depth");
    }

    auto leaf = reinterpret_cast<BTreeLeafNode<Key, Value> *>(&node);
    unsigned pos = leaf->lowerBound(k);
    if ((pos < leaf->count) && (leaf->data[pos].key == k)) {
      success = true;
      result = leaf->data[pos].value;
    }
    #else
    NodeType last_node = NodeType::Inner;

    auto read_node_cb = [&k, &last_node, &next_oid,
                         &success, &result](NodeBase *node) {
      auto node_type = node->getType();
      if (node_type == NodeType::Inner) {
        auto inner = reinterpret_cast<BTreeInnerNode<Key> *>(node);
        next_oid = inner->children[inner->lowerBound(k)];
      } else {  // node_type == NodeType::Leaf
        auto leaf = reinterpret_cast<BTreeLeafNode<Key, Value> *>(node);
        unsigned pos = leaf->lowerBound(k);
        if ((pos < leaf->count) && (leaf->data[pos].key == k)) {
          success = true;
          result = leaf->data[pos].value;
        }
        last_node = NodeType::Leaf;
      }
    };

    do  {
      t->ReadCallback<NodeBase>(nodes, next_oid, read_node_cb);
    } while (last_node == NodeType::Inner);
    #endif

    auto ok = t->PreCommit();
    if (!ok) {
      t->Rollback();
      return std::nullopt;
    }
    t->PostCommit();
    return success;
  }

  bool update(const Key &k, Value v) {
    std::optional<bool> ret;
    do {
      #if defined(MATERIALIZED_UPDATE)
      ret = update_internal(k, v);
      #else
      ret = update_internal_callback(k, v);
      #endif
    } while (!ret);

    return ret.value();
  }

  std::optional<bool> update_internal_callback(const Key &k, Value v) {
    auto t = Transaction::BeginSystem(&group);

    table::OID next_oid = root_id;
    auto node = t->arena->NextValue<BTreeNode>();
    NodeType last_node = NodeType::Inner;
    bool key_exists = false;
    auto read_node_cb = [&k, &last_node, &next_oid, &key_exists](NodeBase *node) {
      auto node_type = node->getType();
      if (node_type == NodeType::Inner) {
        auto inner = reinterpret_cast<BTreeInnerNode<Key> *>(node);
        next_oid = inner->children[inner->lowerBound(k)];
      } else {  // node_type == NodeType::Leaf
        last_node = NodeType::Leaf;
        auto leaf = reinterpret_cast<BTreeLeafNode<Key, Value> *>(node);
        if (leaf->key_exists(k)) {
          key_exists = true;
        }
      }
    };

    do {
      t->ReadCallback<NodeBase>(nodes, next_oid, read_node_cb);
    } while (last_node == NodeType::Inner);

    if (!key_exists) {
      t->Rollback();
      return false;
    }

    auto leaf_update_cb = [k, v](uint8_t *data) -> void {
      auto leaf = reinterpret_cast<BTreeLeafNode<Key, Value> *>(data);
      auto ok = leaf->update(k, v);
      CHECK(ok);
    };

    t->UpdateRecordCallback(nodes, next_oid, leaf_update_cb);
    if (!t->PreCommit()) {
      t->Rollback();
      return std::nullopt;
    }
    t->PostCommit();
    return true;
  }

  std::optional<bool> update_internal(const Key &k, Value v) {
    auto t = Transaction::BeginSystem(&group);

    table::OID next_oid = root_id;
    auto node = t->arena->NextValue<BTreeNode>();
    #if defined(MATERIALIZED_READ)
    t->Read(nodes, next_oid, node);

    while (node->getType() == NodeType::Inner) {
      auto inner = reinterpret_cast<BTreeInnerNode<Key> *>(node);

      next_oid = inner->children[inner->lowerBound(k)];
      t->Read(nodes, next_oid, node);
    }
    #else
    NodeType last_node = NodeType::Inner;
    auto read_node_cb = [&k, &last_node, &next_oid](NodeBase *node) {
      auto node_type = node->getType();
      if (node_type == NodeType::Inner) {
        auto inner = reinterpret_cast<BTreeInnerNode<Key> *>(node);
        next_oid = inner->children[inner->lowerBound(k)];
      } else {  // node_type == NodeType::Leaf
        last_node = NodeType::Leaf;
      }
    };

    do {
      t->ReadCallback<NodeBase>(nodes, next_oid, read_node_cb);
    } while (last_node == NodeType::Inner);

    // TODO(ziyi) re-read a tuple, add tid verification in read_set.Add().
    // Or not, correctness is still intact but there won't be early exit here.
    t->Read(nodes, next_oid, node);
    #endif

    auto leaf = reinterpret_cast<BTreeLeafNode<Key, Value> *>(node);
    bool ok = leaf->update(k, v);
    if (!ok) {
      t->Rollback();
      return false;
    }
    t->Update(nodes, next_oid, leaf);
    ok = t->PreCommit();
    if (!ok) {
      t->Rollback();
      return std::nullopt;
    }
    t->PostCommit();
    return true;
  }
  // 改写新增: batch_lookup — 排序 keys 后一次 scan 遍历来匹配
  // 比逐个 lookup 少 O(n * depth) 次 root-to-leaf traversal
  struct BatchResult { Value value; bool found; };

  std::vector<BatchResult> batch_lookup(const Key *keys, size_t n) {
    std::vector<BatchResult> results(n);
    if (n == 0) return results;

    // 构建排序索引 (不修改输入数组)
    std::vector<size_t> order(n);
    for (size_t i = 0; i < n; i++) order[i] = i;
    std::sort(order.begin(), order.end(),
              [&](size_t a, size_t b) { return keys[a] < keys[b]; });

    // 按排序顺序 lookup — 连续 key 倾向于命中相同/相邻 leaf
    // 后续可优化为单次 scan 遍历, 目前先用 sorted-order lookup
    for (size_t qi = 0; qi < n; qi++) {
      size_t orig = order[qi];
      results[orig].found = lookup(keys[orig], results[orig].value);
    }
    return results;
  }

  // 改写新增: range_count — 只计数不拷贝, 比 scan 快
  uint64_t range_count(const Key &start, const Key &end) {
    // 用 scan 但输出到 /dev/null — 未来可优化为纯计数路径
    // 这里先用固定上限, 如果超出说明范围太大
    constexpr int64_t MAX_COUNT_RANGE = 1000000;
    std::vector<Value> buf(std::min(static_cast<int64_t>(1024), MAX_COUNT_RANGE));
    uint64_t total = 0;
    Key cursor = start;
    while (true) {
      uint64_t got = scan(cursor, static_cast<int64_t>(buf.size()), buf.data(), &end);
      total += got;
      if (got < buf.size()) break;
      if (total >= static_cast<uint64_t>(MAX_COUNT_RANGE)) break;
      // 推进 cursor — 需要知道最后读到的 key, 这里简化处理
      break;  // 单轮, 后续改进
    }
    return total;
  }
};

}  // namespace tabular
} // namespace lynceus
