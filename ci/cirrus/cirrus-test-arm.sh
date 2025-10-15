pytest -m "(not slow)" --durations=25 --cov=randomgen --cov-branch --cov-report xml:coverage.xml --cov-report term randomgen/tests
apt-get install -y curl
curl -Os https://uploader.codecov.io/v0.8.0/aarch64/codecov
chmod +x codecov
./codecov -f coverage.xml -F adder -F subtractor
