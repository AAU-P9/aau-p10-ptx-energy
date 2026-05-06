CARGO    ?= cargo
PREFIX   ?= /usr/local
BINDIR   ?= $(PREFIX)/bin
BIN_NAME ?= ptx-analyser

.PHONY: build release install uninstall clean

build:
	$(CARGO) build

release:
	$(CARGO) build --release

install:
	$(CARGO) install --path .
	@echo "Installed $(BIN_NAME) to $$(dirname $$(which $(BIN_NAME)))"

uninstall:
	$(CARGO) uninstall $(BIN_NAME)
	@echo "Removed $(BIN_NAME)"

clean:
	$(CARGO) clean
