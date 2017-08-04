default: run

deps:
	go get github.com/pkg/profile
	go get gopkg.in/urfave/cli.v1

build:
	go build -o chargoc
.PHONY: build

run: build
	./chargoc

mac:
	go build -o build/mac/chargoc -ldflags "-s -w"

linux:
	GOARCH=amd64 GOOS=linux go build -o build/linux/chargoc -ldflags "-s -w"
