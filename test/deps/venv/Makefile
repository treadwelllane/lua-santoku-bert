results.mk:
	python3 -m venv venv
	source venv/bin/activate && \
		pip3 install torch==2.0.1 faiss-cpu==1.7.4 sentence-transformers==2.2.2
	echo "export TEST_ENVS += deps/venv/venv/bin/activate" > "$@"
