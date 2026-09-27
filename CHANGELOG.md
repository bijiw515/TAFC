# TAFC Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- Initial release of TAFC for FLUX, Wan2.1, and HunyuanVideo
- Trajectory curvature-based caching decision
- First-order extrapolation for cached steps
- Physical time-aware drift estimation
- Adaptive threshold with early-strict, late-relaxed scheduling
- Closed-loop PID control for automatic threshold adjustment
- Evaluation scripts for HPSv3, DrawBench, PyIQA metrics
- Visualization tools for mechanism comparison and velocity analysis
- Comprehensive documentation and installation guide

### Features
- Support for FLUX.1-dev and FLUX.1-schnell (text-to-image)
- Support for Wan2.1-T2V-1.3B and T2V-14B (text-to-video)
- Support for Wan2.1-I2V-14B (image-to-video)
- Support for HunyuanVideo (text-to-video)
- 2-3× speedup with minimal quality degradation
- Configurable parameters for quality-speed tradeoff

## [1.0.0] - 2026-09-27

### Initial Release
- First public release of TAFC
- Core functionality for trajectory-based caching
- Documentation and examples
- Evaluation benchmarks
