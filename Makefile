# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
# Runs a target in every package (PKGS), e.g.  make lib  /  make test qat PKGS=optimizers  /  make lib ARCHS=native
# Targets and variables: <package>/Makefile and build/arch.mk.
PKGS ?= optimizers
TARGETS = all lib test qat bench install clean

$(TARGETS):
	@for p in $(PKGS); do $(MAKE) -C $$p $@ || exit 1; done

.PHONY: $(TARGETS)
