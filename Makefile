default: run

deps:
	go get github.com/pkg/profile
	go get gopkg.in/urfave/cli.v1
	go get gopkg.in/gizak/termui.v2

run:
	go build -o ricur
	./ricur
.PHONY: run