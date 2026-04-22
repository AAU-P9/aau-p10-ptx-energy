CARGO    ?= cargo
PREFIX   ?= /usr/local
BINDIR   ?= $(PREFIX)/bin
BIN_NAME ?= ptx-analyser

.PHONY: build release install uninstall clean

build:
	$(CARGO) build

release:
	$(CARGO) build

install: release
	sudo install -m 755 target/release/$(BIN_NAME) $(BINDIR)/$(BIN_NAME)
	@echo "Installed $(BIN_NAME) to $(BINDIR)/$(BIN_NAME)"

uninstall:
	sudo rm -f $(BINDIR)/$(BIN_NAME)
	@echo "Removed $(BINDIR)/$(BIN_NAME)"

clean:
	$(CARGO) clean
