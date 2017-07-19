package main

import (
	"io/ioutil"
	"os"
)

func readFileContents(filename string) (string, error) {
	buf, err := ioutil.ReadFile(filename)
	if err != nil {
		return "", err
	}
	return string(buf), nil
}

func writeFileContents(filename string, contents []byte) (error) {
	return ioutil.WriteFile(filename, contents, os.ModePerm)
}
