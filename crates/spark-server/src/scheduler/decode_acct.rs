// SPDX-License-Identifier: AGPL-3.0-only

//! Per-request decode-path accounting for the Done-line.
//!
//! A finished request must report how it actually ran: fraction of steps
//! that were plain serial decode vs MTP verify, mean accepted drafts on
//! the MTP steps, and how many depth-regime re-probes the throughput
//! gate fired. These counters are the SSOT the Done-line prints — not a
//! new telemetry product.

/// Per-sequence decode-path counters. Reset implicitly: each `ActiveSeq`
/// is constructed with [`DecodeAcct::default`].
#[derive(Debug, Default, Clone, Copy)]
pub struct DecodeAcct {
    /// Plain `step_decode_only` steps (think-gate, gate serial mode, or
    /// a serial baseline probe).
    pub serial_steps: u64,
    /// MTP (or other speculative) steps that ran a verify/bootstrap.
    pub mtp_steps: u64,
    /// Sum of accepted *drafts* on MTP steps (`emitted.saturating_sub(1)`).
    pub accepted_drafts: u64,
    /// Times `MtpGate::maybe_remeasure` fired a depth-regime change
    /// while this sequence was active.
    pub regime_reprobes: u64,
}

impl DecodeAcct {
    pub fn record_serial(&mut self) {
        self.serial_steps = self.serial_steps.saturating_add(1);
    }

    /// `emitted` is tokens committed this step (1 for bootstrap / full
    /// reject, `1 + accepted_drafts` for a verify).
    pub fn record_mtp_emitted(&mut self, emitted: usize) {
        self.mtp_steps = self.mtp_steps.saturating_add(1);
        self.accepted_drafts = self
            .accepted_drafts
            .saturating_add(emitted.saturating_sub(1) as u64);
    }

    pub fn note_regime_reprobe(&mut self) {
        self.regime_reprobes = self.regime_reprobes.saturating_add(1);
    }

    pub fn total_steps(&self) -> u64 {
        self.serial_steps.saturating_add(self.mtp_steps)
    }

    pub fn serial_frac(&self) -> f64 {
        let n = self.total_steps();
        if n == 0 {
            0.0
        } else {
            self.serial_steps as f64 / n as f64
        }
    }

    pub fn mtp_frac(&self) -> f64 {
        let n = self.total_steps();
        if n == 0 {
            0.0
        } else {
            self.mtp_steps as f64 / n as f64
        }
    }

    /// Mean accepted drafts per MTP step. 0.0 when no MTP step ran.
    pub fn mean_accepted(&self) -> f64 {
        if self.mtp_steps == 0 {
            0.0
        } else {
            self.accepted_drafts as f64 / self.mtp_steps as f64
        }
    }

    /// Structured suffix appended to the existing Done-line.
    pub fn done_suffix(&self) -> String {
        format!(
            "serial={:.2} mtp={:.2} mean_accepted={:.2} regime_reprobes={}",
            self.serial_frac(),
            self.mtp_frac(),
            self.mean_accepted(),
            self.regime_reprobes
        )
    }

    /// The request-finished log. Lives here so `lifecycle.rs` stays under
    /// the 500-LoC cap.
    pub fn log_done(n: usize, reason: &str, tps: f64, ttft_ms: f64, acct: &Self) {
        tracing::info!(
            "Done: {n} tokens ({reason}) {tps:.1} tok/s, TTFT={ttft_ms:.1}ms, {}",
            acct.done_suffix()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_acct_is_zeros() {
        let a = DecodeAcct::default();
        assert_eq!(
            a.done_suffix(),
            "serial=0.00 mtp=0.00 mean_accepted=0.00 regime_reprobes=0"
        );
    }

    #[test]
    fn thinking_serial_run_is_all_serial() {
        let mut a = DecodeAcct::default();
        for _ in 0..300 {
            a.record_serial();
        }
        assert!((a.serial_frac() - 1.0).abs() < 1e-9);
        assert_eq!(a.mtp_frac(), 0.0);
        assert_eq!(a.mean_accepted(), 0.0);
        assert!(a.done_suffix().contains("serial=1.00"));
        assert!(a.done_suffix().contains("mtp=0.00"));
    }

    #[test]
    fn mtp_run_reports_mean_accepted() {
        let mut a = DecodeAcct::default();
        // 10 verify steps, 1.3 drafts accepted on average (13 drafts / 10).
        for _ in 0..7 {
            a.record_mtp_emitted(2); // 1 draft
        }
        for _ in 0..3 {
            a.record_mtp_emitted(3); // 2 drafts
        }
        assert!((a.mtp_frac() - 1.0).abs() < 1e-9);
        assert!((a.mean_accepted() - 1.3).abs() < 1e-9);
        a.note_regime_reprobe();
        assert!(a.done_suffix().contains("mean_accepted=1.30"));
        assert!(a.done_suffix().contains("regime_reprobes=1"));
    }
}
