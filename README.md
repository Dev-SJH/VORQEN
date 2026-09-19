# VORQEN

> A small language model built from scratch by a middle schooler.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## 📖 소개

VORQEN은 중학생이 **밑바닥부터 직접 만드는** 언어 모델 프로젝트입니다.
거대 기업의 GPU 없이, 무료 컴퓨팅 자원(Google Colab, Kaggle)만으로
작은 언어 모델을 학습시키고, 점진적으로 발전시키는 것을 목표로 합니다.

## 🎯 목표

- **단기**: 트랜스포머 원리 완전 이해 + 미니 GPT 구현
- **중기**: 한국어 데이터로 10M~50M 파라미터 모델 학습
- **장기**: 오픈소스 소형 모델을 파인튜닝하여 실용적인 챗봇 배포

## 🗺️ 로드맵

- [x] Phase 0: GitHub 레포 세팅
- [ ] Phase 1: 미니 GPT (1M) — 트랜스포머 밑바닥 구현
- [ ] Phase 2: 한국어 VORQEN v0.1 (10M) — 한국어 데이터 학습
- [ ] Phase 3: VORQEN 챗봇 — LoRA 파인튜닝 + Hugging Face 배포
- [ ] Phase 4: VORQEN Pro — 도메인 특화 모델

## 🛠️ 설치

```bash
git clone https://github.com/Dev-SJH/VORQEN.git
cd VORQEN
pip install -r requirements.txt