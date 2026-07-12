"""测试风险规则引擎"""

from src.risk.rule_engine import RiskRuleEngine
from src.chunkers.legal_chunker import Chunk


class TestRuleEngine:
    def test_load_rules(self):
        engine = RiskRuleEngine()
        assert len(engine._rules) >= 10

    def test_keyword_hit(self):
        engine = RiskRuleEngine()
        chunk = Chunk(
            chunk_id="x",
            doc_id="test",
            text="违约金为合同总价的80%",
            chunk_index=0,
            clause_ref="test",
            start_char=0,
            end_char=50,
        )
        hits = engine.scan("test", [chunk])
        # "违约金" 应命中 R001
        assert any(h.rule.id == "R001" for h in hits)

    def test_no_hit(self):
        engine = RiskRuleEngine()
        chunk = Chunk(
            chunk_id="x",
            doc_id="test",
            text="本合同自双方签字之日起生效",
            chunk_index=0,
            clause_ref="test",
            start_char=0,
            end_char=50,
        )
        hits = engine.scan("test", [chunk])
        assert hits == []
