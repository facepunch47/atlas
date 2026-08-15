// SPDX-License-Identifier: AGPL-3.0-only

//! MTP dispatch eligibility — the think-gate predicate.
//!
//! Extracted so the scheduler loop and the unit tests share one function.
//! A thinking-on request used to fail this predicate for the entire
//! `<think>` span (`inside_thinking && !dflash_spec_think`), which sent
//! every think token through `step_decode_only`. That is a different
//! machine from thinking-off (MTP K-verify) and is the 22→6 tok/s
//! collapse on Qwen3.8-27B (recipe serve is thinking ON:
//! `[behavior].thinking_default = true`, `max_thinking_budget = 2048`).
//! The throughput-arbitrated gate (#337 / #344 / #242 / d6171c4) is
//! already shipped and is not this predicate — `maybe_remeasure` only
//! runs once a sequence is already on the MTP branch.
//!
//! Standard MTP verify still runs
//! [`crate::scheduler::logit_processors::forced_think_end`], so
//! thinking-budget injection stays on the verify path. DFlash raw-argmax
//! does not, and stays serial-in-think unless `ATLAS_DFLASH_SPEC_THINK=1`.
//!
//! #517 (90% think cap from raw `max_tokens` before `--tool-max-tokens`
//! shrink) can let a tool turn think for ~900 tokens. That is think
//! *length*, not the tok/s collapse — cite, do not fix here.

/// Whether this sequence may take the MTP verify path on this step.
pub(super) fn mtp_spec_eligible(
    inside_thinking: bool,
    post_think_emitted: u32,
    output_len: u32,
    suppress_tool_call: bool,
    disable_mtp: bool,
    spec_think: bool,
    resume_guard: u32,
    dflash_raw_argmax: bool,
) -> bool {
    if suppress_tool_call || disable_mtp {
        return false;
    }
    // DFlash raw-argmax: thinking-budget forced-end is not on that path.
    // Stay serial in think unless SPEC_THINK is explicitly on.
    if dflash_raw_argmax && !spec_think {
        return !inside_thinking && post_think_emitted >= resume_guard;
    }
    // Standard MTP (and DFlash+SPEC_THINK): speculate in think and after.
    // The resume guard still serial-decodes the entry window — T=0 flips
    // concentrate at spec entry (sequence start) and post-`</think>` resume.
    if inside_thinking {
        output_len >= resume_guard
    } else {
        post_think_emitted >= resume_guard
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_mtp_thinks() {
        // thinking-on, guard off (the shipped default): MTP from token 0.
        assert!(mtp_spec_eligible(true, 0, 0, false, false, false, 0, false));
        assert!(mtp_spec_eligible(
            true, 0, 50, false, false, false, 0, false
        ));
    }

    #[test]
    fn standard_mtp_respects_entry_guard_in_think() {
        assert!(!mtp_spec_eligible(
            true, 0, 3, false, false, false, 7, false
        ));
        assert!(mtp_spec_eligible(true, 0, 7, false, false, false, 7, false));
    }

    #[test]
    fn standard_mtp_respects_post_think_resume_guard() {
        assert!(!mtp_spec_eligible(
            false, 0, 300, false, false, false, 7, false
        ));
        assert!(mtp_spec_eligible(
            false, 7, 300, false, false, false, 7, false
        ));
    }

    #[test]
    fn thinking_off_is_eligible_at_guard_zero() {
        assert!(mtp_spec_eligible(
            false, 0, 0, false, false, false, 0, false
        ));
    }

    #[test]
    fn dflash_raw_argmax_stays_serial_in_think() {
        assert!(!mtp_spec_eligible(
            true, 0, 50, false, false, false, 0, true
        ));
        // After think, DFlash may spec (guard 0).
        assert!(mtp_spec_eligible(
            false, 0, 50, false, false, false, 0, true
        ));
    }

    #[test]
    fn dflash_spec_think_opts_in() {
        assert!(mtp_spec_eligible(true, 0, 0, false, false, true, 0, true));
        assert!(!mtp_spec_eligible(true, 0, 3, false, false, true, 7, true));
        assert!(mtp_spec_eligible(true, 0, 7, false, false, true, 7, true));
    }

    #[test]
    fn suppress_and_disable_always_block() {
        assert!(!mtp_spec_eligible(
            true, 0, 10, true, false, false, 0, false
        ));
        assert!(!mtp_spec_eligible(
            true, 0, 10, false, true, false, 0, false
        ));
        assert!(!mtp_spec_eligible(false, 0, 10, true, false, true, 0, true));
    }
}
