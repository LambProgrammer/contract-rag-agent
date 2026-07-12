"""测试条款分块器"""

from src.chunkers.legal_chunker import LegalClauseChunker


class TestLegalChunker:
    def test_split_by_clause_number(self):
        chunker = LegalClauseChunker(min_chunk_size=10, max_chunk_size=500)
        text = "## 第一条 违约责任\n违约金为合同总价的20%，应于违约后10日内支付。\n\n## 第二条 费用支付\n付款后不退。"
        chunks = chunker.chunk(text, "test")
        assert len(chunks) >= 2

    def test_extract_appendix_ref(self):
        chunker = LegalClauseChunker()
        assert "附件一：服务费用明细" in chunker._extract_clause_ref(
            "附件一：服务费用明细\n\n1. 住宿费每天200元"
        )

    def test_merge_short_chunks(self):
        chunker = LegalClauseChunker(min_chunk_size=50, max_chunk_size=1500)
        short = ["## 第一条", "违约金20%。" * 10]
        merged = chunker._merge_short_chunks(short)
        assert len(merged) == 1

    def test_empty_text(self):
        chunker = LegalClauseChunker()
        chunks = chunker.chunk("", "test")
        assert chunks == []
