default: run

deps:
	go get github.com/pkg/profile
	go get gopkg.in/urfave/cli.v1

run:
	go build -o ricur
	./ricur
.PHONY: run