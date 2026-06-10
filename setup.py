from setuptools import find_packages, setup


setup(
	name="neurotrain",
	version="0.1.0",
	description="Minimal educational deep learning framework built step by step.",
	long_description=(
		"NeuroTrain is a lightweight educational framework to understand "
		"how tensors, layers, losses, optimizers, and autograd work together."
	),
	long_description_content_type="text/plain",
	author="NeuroTrain",
	python_requires=">=3.10",
	packages=find_packages(),
	include_package_data=True,
	install_requires=[
		"numpy>=1.24,<3.0",
	],
)
