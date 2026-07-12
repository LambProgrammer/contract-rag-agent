"""测试稀疏向量生成"""

from src.retrieval.sparse_retriever import tokenize, token_to_id, text_to_sparse_vector


class TestSparseVector:
    def test_tokenize_chinese(self):
        tokens = tokenize("违约责任怎么计算")
        assert len(tokens) >= 2
        assert all(len(t) > 1 for t in tokens)

    def test_token_to_id_range(self):
        tid = token_to_id("违约金")
        assert 0 <= tid < 1_000_000

    def test_text_to_sparse_vector(self):
        indices, values = text_to_sparse_vector("违约责任怎么计算")
        assert len(indices) > 0
        assert len(indices) == len(values)
        assert all(0.0 <= v <= 1.0 for v in values)
        # 不应有重复 indices（已合并 hash 冲突）
        assert len(indices) == len(set(indices))

    def test_empty_text(self):
        indices, values = text_to_sparse_vector("")
        assert indices == []
        assert values == []
