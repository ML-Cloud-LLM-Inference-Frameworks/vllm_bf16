import unittest

from common.prometheus_utils import (
    diff_prometheus_samples,
    extract_kv_cache_usage_perc_max,
    openai_v1_base_to_metrics_url,
    summarize_vllm_samples,
)


class PrometheusUtilsTests(unittest.TestCase):
    def test_openai_v1_base_to_metrics_url(self):
        self.assertEqual(
            openai_v1_base_to_metrics_url("http://127.0.0.1:8000/v1"),
            "http://127.0.0.1:8000/metrics",
        )
        self.assertEqual(
            openai_v1_base_to_metrics_url("http://127.0.0.1:8000/custom/v1"),
            "http://127.0.0.1:8000/custom/metrics",
        )

    def test_extract_kv_cache_usage_prefers_supported_names(self):
        samples = {
            'vllm:kv_cache_usage_perc{engine="0"}': 0.25,
            'vllm:kv_cache_usage_perc{engine="1"}': 0.5,
        }
        self.assertEqual(extract_kv_cache_usage_perc_max(samples), 0.5)

    def test_diff_and_summary_support_delta_based_histograms(self):
        before = {
            'vllm:time_to_first_token_seconds_sum{engine="0"}': 10.0,
            'vllm:time_to_first_token_seconds_count{engine="0"}': 20,
            'vllm:request_queue_time_seconds_sum{engine="0"}': 2.0,
            'vllm:request_queue_time_seconds_count{engine="0"}': 20,
            'vllm:prefix_cache_hits_total{engine="0"}': 100,
            'vllm:prefix_cache_queries_total{engine="0"}': 200,
        }
        after = {
            'vllm:time_to_first_token_seconds_sum{engine="0"}': 16.0,
            'vllm:time_to_first_token_seconds_count{engine="0"}': 24,
            'vllm:request_queue_time_seconds_sum{engine="0"}': 3.0,
            'vllm:request_queue_time_seconds_count{engine="0"}': 24,
            'vllm:prefix_cache_hits_total{engine="0"}': 140,
            'vllm:prefix_cache_queries_total{engine="0"}': 260,
            'vllm:kv_cache_usage_perc{engine="0"}': 0.75,
        }

        delta = diff_prometheus_samples(after, before)
        summary = summarize_vllm_samples(delta)

        self.assertAlmostEqual(summary["ttft_mean_s"], 1.5)
        self.assertEqual(summary["ttft_mean_s_count"], 4)
        self.assertAlmostEqual(summary["queue_time_mean_s"], 0.25)
        self.assertEqual(summary["prefix_cache_hits"], 40)
        self.assertEqual(summary["prefix_cache_queries"], 60)
        self.assertAlmostEqual(summary["prefix_cache_hit_rate"], 40 / 60)


if __name__ == "__main__":
    unittest.main()
