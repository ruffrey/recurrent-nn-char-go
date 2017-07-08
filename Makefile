default: run

deps:
	go get github.com/pkg/profile
	go get gopkg.in/urfave/cli.v1

build:
	go build -o ricur

run: build
	./ricur
.PHONY: run

prod:
	go build -o ricur -ldflags "-s -w"
