import unittest

from app.contracts import (
    AgentHandoff,
    ReviewFinding,
    RoleId,
    StageContract,
    TokenUsage,
)


class ContractTests(unittest.TestCase):
    def test_handoff_round_trip(self):
        contract = StageContract(
            run_id="RUN-001",
            stage_id="stage-001",
            objective="Build the foundation",
            scope=("app/",),
            acceptance_criteria=("Tests pass",),
            verification_commands=("python -m unittest",),
        )
        finding = ReviewFinding(
            finding_id="R-001",
            severity="medium",
            file="app/example.py",
            line=3,
            evidence="Missing guard",
            required_change="Add the guard",
        )
        handoff = AgentHandoff(
            contract=contract,
            from_role=RoleId.REVIEW,
            to_role=RoleId.IMPROVEMENT,
            summary="One supported issue",
            findings=(finding,),
            usage=TokenUsage(input_tokens=100, output_tokens=20),
        )
        restored = AgentHandoff.from_dict(handoff.to_dict())
        self.assertEqual(handoff, restored)
        self.assertEqual(120, restored.usage.total_tokens)
        self.assertEqual("implementation", restored.findings[0].category)

    def test_contract_requires_verifiable_scope(self):
        with self.assertRaises(ValueError):
            StageContract(
                run_id="RUN-001",
                stage_id="stage-001",
                objective="Incomplete",
                scope=(),
                acceptance_criteria=("Done",),
                verification_commands=("test",),
            )

    def test_contract_rejects_ambiguous_or_escaping_stage_scope(self):
        for scope in ((".",), ("../outside.py",), ("C:\\outside.py",), ("src/*.py",)):
            with self.subTest(scope=scope), self.assertRaises(ValueError):
                StageContract(
                    run_id="RUN-001",
                    stage_id="stage-001",
                    objective="Unsafe scope",
                    scope=scope,
                    acceptance_criteria=("Done",),
                    verification_commands=("py -m unittest",),
                )

    def test_handoff_cannot_target_same_role(self):
        contract = StageContract(
            run_id="RUN-001",
            stage_id="stage-001",
            objective="Test",
            scope=("app/",),
            acceptance_criteria=("Done",),
            verification_commands=("test",),
        )
        with self.assertRaises(ValueError):
            AgentHandoff(
                contract=contract,
                from_role=RoleId.REVIEW,
                to_role=RoleId.REVIEW,
                summary="Invalid",
            )

    def test_handoff_cannot_skip_or_reverse_role_order(self):
        contract = StageContract(
            run_id="RUN-001",
            stage_id="stage-001",
            objective="Test",
            scope=("app/",),
            acceptance_criteria=("Done",),
            verification_commands=("py -m unittest",),
        )
        with self.assertRaises(ValueError):
            AgentHandoff(
                contract=contract,
                from_role=RoleId.DEVELOPMENT,
                to_role=RoleId.IMPROVEMENT,
                summary="Skipped review",
            )

    def test_review_routes_design_and_implementation_findings_to_correct_roles(self):
        contract = StageContract(
            run_id="RUN-001",
            stage_id="stage-001",
            objective="Test",
            scope=("app/",),
            acceptance_criteria=("Done",),
            verification_commands=("py -m unittest",),
        )
        design = ReviewFinding(
            finding_id="D-001",
            severity="high",
            category="design",
            evidence="계약과 접근 방향이 다름",
            required_change="접근 방향 재설계",
        )
        implementation = ReviewFinding(
            finding_id="I-001",
            severity="medium",
            category="implementation",
            evidence="경계 검사 누락",
            required_change="경계 검사 추가",
        )

        AgentHandoff(
            contract=contract,
            from_role=RoleId.REVIEW,
            to_role=RoleId.DEVELOPMENT,
            summary="설계 재작업 필요",
            findings=(design,),
        )
        with self.assertRaises(ValueError):
            AgentHandoff(
                contract=contract,
                from_role=RoleId.REVIEW,
                to_role=RoleId.IMPROVEMENT,
                summary="잘못된 라우팅",
                findings=(design,),
            )
        with self.assertRaises(ValueError):
            AgentHandoff(
                contract=contract,
                from_role=RoleId.REVIEW,
                to_role=RoleId.DEVELOPMENT,
                summary="잘못된 라우팅",
                findings=(implementation,),
            )


if __name__ == "__main__":
    unittest.main()
