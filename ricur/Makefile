default: run

deps:
	go get github.com/pkg/profile
	go get gopkg.in/urfave/cli.v1

build:
	go build -o ricur
.PHONY: build

run: build
	./ricur

mac:
	go build -o build/mac/ricur -ldflags "-s -w"

linux:
	GOARCH=amd64 GOOS=linux go build -o build/linux/ricur -ldflags "-s -w"
