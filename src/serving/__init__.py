# -*- coding: utf-8 -*-
"""Serving
Real-time recommendation serving for the Toronto Data Platform (TDP).

This package replaces a managed-endpoint approach with a self-hosted **FastAPI**
service that loads a model registered by the training pipeline and serves ranked
program slates over HTTP. It is packaged as a container image (``Docker/
Dockerfile.serving``) and deployed to an **Amazon EKS** cluster behind an AWS
ALB ingress, autoscaled with a Horizontal Pod Autoscaler (see ``deploy/k8s``).
"""
from .predictor import Predictor
