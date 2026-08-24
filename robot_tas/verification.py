from __future__ import annotations

import logging
from pathlib import Path

from robot_tas.api.base import MultimodalClient
from robot_tas.cache import ensure_dir, read_json, write_json
from robot_tas.schemas import LocalBoundaryProposal, SampledFrame, VerifiedBoundary
from robot_tas.windows import boundary_neighborhood


LOGGER = logging.getLogger(__name__)


def run_boundary_verification(
    sampled_frames: list[SampledFrame],
    proposals: list[LocalBoundaryProposal],
    client: MultimodalClient,
    prompt_text: str,
    prompt_version: str,
    output_dir: Path,
    verification_radius: int,
    force: bool = False,
) -> list[VerifiedBoundary]:
    """Verify each proposed boundary independently."""

    stage_path = output_dir / "verified_boundaries.json"
    if stage_path.exists() and not force:
        return [VerifiedBoundary.model_validate(item) for item in read_json(stage_path)]

    cache_dir = ensure_dir(output_dir / "cache" / "verified_boundaries")
    raw_dir = ensure_dir(output_dir / "raw_api" / "verified_boundaries")
    verified: list[VerifiedBoundary] = []

    for proposal in proposals:
        for boundary_index, candidate in enumerate(proposal.boundary_candidates):
            proposal_id = f"window_{proposal.window_id}_boundary_{boundary_index}"
            item_path = cache_dir / f"{proposal_id}.json"
            if item_path.exists() and not force:
                result = VerifiedBoundary.model_validate(read_json(item_path))
            else:
                neighborhood = boundary_neighborhood(
                    sampled_frames=sampled_frames,
                    boundary_sample_index=candidate.boundary_sample_index,
                    radius=verification_radius,
                )
                call = client.verify_boundary(
                    proposal_id=proposal_id,
                    candidate=candidate,
                    neighborhood=neighborhood,
                    prompt_text=prompt_text,
                    prompt_version=prompt_version,
                )
                result = call.parsed.model_copy(update={"window_id": proposal.window_id})
                write_json(item_path, result.model_dump(mode="json"))
                write_json(
                    raw_dir / f"{proposal_id}.json",
                    {"request": call.raw_request, "response": call.raw_response, "cache_key": call.cache_key},
                )
                LOGGER.info("Verified proposal %s with status=%s", proposal_id, result.status)
            verified.append(result)

    write_json(stage_path, [item.model_dump(mode="json") for item in verified])
    return verified

