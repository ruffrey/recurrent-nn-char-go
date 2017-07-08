default: run

deps:
	go get github.com/pkg/profile

run:
	go build -o ricur
	./ricur
.PHONY: run