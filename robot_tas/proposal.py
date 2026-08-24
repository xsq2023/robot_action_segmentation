from __future__ import annotations

import logging
from pathlib import Path

from robot_tas.api.base import MultimodalClient
from robot_tas.cache import ensure_dir, read_json, write_json
from robot_tas.schemas import LocalBoundaryProposal, Window


LOGGER = logging.getLogger(__name__)


def run_local_boundary_proposals(
    windows: list[Window],
    client: MultimodalClient,
    prompt_text: str,
    prompt_version: str,
    output_dir: Path,
    force: bool = False,
) -> list[LocalBoundaryProposal]:
    """Run or reuse local boundary proposals for each window."""

    stage_path = output_dir / "local_proposals.json"
    if stage_path.exists() and not force:
        return [LocalBoundaryProposal.model_validate(item) for item in read_json(stage_path)]

    cache_dir = ensure_dir(output_dir / "cache" / "local_proposals")
    raw_dir = ensure_dir(output_dir / "raw_api" / "local_proposals")
    proposals: list[LocalBoundaryProposal] = []

    for window in windows:
        item_path = cache_dir / f"window_{window.window_id:04d}.json"
        if item_path.exists() and not force:
            proposal = LocalBoundaryProposal.model_validate(read_json(item_path))
        else:
            result = client.propose_boundaries(window=window, prompt_text=prompt_text, prompt_version=prompt_version)
            proposal = result.parsed
            write_json(item_path, proposal.model_dump(mode="json"))
            write_json(
                raw_dir / f"window_{window.window_id:04d}.json",
                {"request": result.raw_request, "response": result.raw_response, "cache_key": result.cache_key},
            )
            LOGGER.info(
                "Proposed %s boundaries for window %s",
                len(proposal.boundary_candidates),
                window.window_id,
            )
        proposals.append(proposal)

    write_json(stage_path, [proposal.model_dump(mode="json") for proposal in proposals])
    return proposals

