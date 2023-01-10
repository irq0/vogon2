#!/bin/bash

source "$(dirname "$0")/../vogon.sh"

(
    cd "$VOGON_TEST_REPOS" || exit

    GIT_BRANCH="$(git branch --show-current)"

    if [ -n "$GIT_BRANCH" ]; then
        vogon_result "git-branch" "$GIT_BRANCH" " "
        vogon_result "git-commit-hash" "$(git log --pretty=format:%H -n 1)" " "
        vogon_result "git-commit-subject" "$(git log --pretty=format:%s -n 1)" " "
        vogon_result "git-commit-data" "$(git log --pretty=format:%ci -n 1)"  " "
    fi

    vogon_time ant clean dist
)
