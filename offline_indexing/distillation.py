"""View distillation and refinement."""

from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger


class ViewDistillation:
    """Distillation and refinement of blind views."""

    def __init__(self, llm_interface, use_patch_refinement: bool = True):
        """
        Initialize view distillation.

        Args:
            llm_interface: LLM interface for refinement
            use_patch_refinement: Whether to use patch-guided refinement
        """
        self.llm = llm_interface
        self.use_patch_refinement = use_patch_refinement
        self._prompts = self._load_prompts()

    def _default_prompt_dir(self) -> Path:
        """
        Get default distillation prompt directory.

        Returns:
            Path to prompts/offline/distill/
        """
        return Path(__file__).resolve().parent.parent / "prompts" / "offline" / "distill"

    def _read_prompt(self, path: Path) -> Optional[str]:
        """
        Read a prompt template from file.

        Args:
            path: Prompt file path

        Returns:
            Prompt content if exists, otherwise None
        """
        try:
            if not path.exists():
                return None
            content = path.read_text(encoding="utf-8")
            return content.strip() or None
        except Exception as e:
            logger.warning(f"Failed to read prompt file {path}: {e}")
            return None

    def _load_prompts(self) -> Dict[str, str]:
        """
        Load distillation prompt templates.

        Returns:
            Dict with keys: with_patch, without_patch, quality_check
        """
        prompt_dir = self._default_prompt_dir()
        prompts: Dict[str, str] = {}

        with_patch = self._read_prompt(prompt_dir / "with_patch.txt")
        without_patch = self._read_prompt(prompt_dir / "without_patch.txt")
        
        # Quality check prompt in offline/ directory
        quality_check = self._read_prompt(
            prompt_dir.parent / "quality_check.txt"
        )

        if with_patch is not None:
            prompts["with_patch"] = with_patch
        if without_patch is not None:
            prompts["without_patch"] = without_patch
        if quality_check is not None:
            prompts["quality_check"] = quality_check

        return prompts

    def distill_view(
        self,
        blind_view: str,
        buggy_code: str,
        view_type: str,
        patch: Optional[str] = None,
    ) -> str:
        """
        Distill a single blind view.

        Args:
            blind_view: Blind view RCA text
            buggy_code: Original buggy code
            view_type: Type of view
            patch: Optional ground truth patch for refinement

        Returns:
            Distilled view text
        """
        logger.debug(f"Distilling {view_type} view")

        if self.use_patch_refinement and patch:
            return self._refine_with_patch(blind_view, buggy_code, patch, view_type)
        else:
            return self._distill_without_patch(blind_view, view_type)

    def _refine_with_patch(
        self, blind_view: str, buggy_code: str, patch: str, view_type: str
    ) -> str:
        """
        Refine view using ground truth patch.

        Args:
            blind_view: Blind view RCA
            buggy_code: Buggy code
            patch: Ground truth patch
            view_type: View type

        Returns:
            Refined view
        """
        template = self._prompts.get("with_patch")
        if template:
            prompt = template.format(
                view_type=view_type,
                blind_view=blind_view,
                buggy_code=buggy_code,
                patch=patch,
            )
        else:
            prompt = f"""Given the blind analysis, buggy code, and the actual patch, refine the analysis to be more accurate and concise.

View Type: {view_type}

Blind Analysis:
{blind_view}

Buggy Code:
{buggy_code}

Actual Patch:
{patch}

Provide a refined, distilled analysis focusing on the most critical aspects:"""

        try:
            refined = self.llm.generate(prompt, max_tokens=512, temperature=0.2)
            return refined
        except Exception as e:
            logger.error(f"Patch-guided refinement failed: {e}")
            return blind_view  # Fallback to blind view

    def _distill_without_patch(self, blind_view: str, view_type: str) -> str:
        """
        Distill view without patch (compression/cleaning).

        Args:
            blind_view: Blind view RCA
            view_type: View type

        Returns:
            Distilled view
        """
        template = self._prompts.get("without_patch")
        if template:
            prompt = template.format(view_type=view_type, blind_view=blind_view)
        else:
            prompt = f"""Distill and compress the following analysis, keeping only the most important insights.
Remove redundancy and focus on actionable information.

View Type: {view_type}

Analysis:
{blind_view}

Provide a concise distilled version:"""

        try:
            distilled = self.llm.generate(prompt, max_tokens=512, temperature=0.1)
            return distilled
        except Exception as e:
            logger.error(f"Distillation failed: {e}")
            return blind_view  # Fallback

    def batch_distill(
        self,
        blind_views: List[str],
        buggy_codes: List[str],
        view_type: str,
        patches: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Batch distill multiple views.

        Args:
            blind_views: List of blind view texts
            buggy_codes: List of buggy codes
            view_type: View type
            patches: Optional list of patches

        Returns:
            List of distilled views
        """
        logger.info(f"Batch distilling {len(blind_views)} {view_type} views")

        if patches is None:
            patches = [None] * len(blind_views)

        distilled = []
        for i, (view, code, patch) in enumerate(zip(blind_views, buggy_codes, patches)):
            if i % 10 == 0:
                logger.info(f"Progress: {i}/{len(blind_views)}")

            distilled_view = self.distill_view(view, code, view_type, patch)
            distilled.append(distilled_view)

        return distilled

    def extract_key_facts(self, distilled_view: str) -> List[str]:
        """
        Extract key facts from distilled view.

        Args:
            distilled_view: Distilled view text

        Returns:
            List of key facts
        """
        # Simple extraction: split by newlines/bullet points
        # In production, use more sophisticated NLP
        facts = []
        lines = distilled_view.split("\n")

        for line in lines:
            line = line.strip()
            if line and len(line) > 10:  # Filter very short lines
                # Remove bullet points and numbering
                line = line.lstrip("•-*123456789. ")
                if line:
                    facts.append(line)

        return facts

    def assess_quality(
        self,
        blind_distilled_view: str,
        buggy_code: str,
        view_type: str,
    ) -> Dict:
        """
        Assess quality of blind distilled view (without seeing ground truth patch).
        
        Determines if the blind distillation is accurate enough to use directly,
        or if patch-guided refinement is needed.

        Args:
            blind_distilled_view: Distilled view from blind analysis
            buggy_code: Original buggy code
            view_type: Type of view

        Returns:
            Dict with keys:
                - accurate (bool): Whether the blind distillation is accurate
                - confidence (float): Confidence score 0.0-1.0
                - reasoning (str): Explanation
                - missing_aspects (list): Issues if not accurate
        """
        template = self._prompts.get("quality_check")
        
        if template is None:
            logger.warning("Quality check prompt not found, defaulting to accurate=False")
            return {
                "accurate": False,
                "confidence": 0.0,
                "reasoning": "Quality check prompt not available",
                "missing_aspects": ["Unable to assess"],
            }

        prompt = template.format(
            view_type=view_type,
            buggy_code=buggy_code,
            blind_distilled_view=blind_distilled_view,
        )

        try:
            response = self.llm.generate(prompt, max_tokens=512, temperature=0.1)
            
            # Parse JSON response
            import json
            import re
            
            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON directly
                json_match = re.search(r'\{.*?\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    json_str = response
            
            result = json.loads(json_str)
            
            # Validate required fields
            if "accurate" not in result:
                logger.warning("Quality assessment missing 'accurate' field")
                result["accurate"] = False
            
            result.setdefault("confidence", 0.5)
            result.setdefault("reasoning", "")
            result.setdefault("missing_aspects", [])
            
            logger.debug(
                f"Quality assessment for {view_type}: "
                f"accurate={result['accurate']}, confidence={result['confidence']:.2f}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            # Conservative fallback: assume not accurate
            return {
                "accurate": False,
                "confidence": 0.0,
                "reasoning": f"Assessment error: {str(e)}",
                "missing_aspects": ["Assessment failed"],
            }

    def merge_views(self, views: Dict[str, str]) -> str:
        """
        Merge multiple view types into unified representation.

        Args:
            views: Dictionary mapping view types to distilled views

        Returns:
            Merged view representation
        """
        merged_parts = []

        for view_type, view_content in views.items():
            merged_parts.append(f"[{view_type.upper()}]")
            merged_parts.append(view_content)
            merged_parts.append("")  # Blank line

        return "\n".join(merged_parts)
