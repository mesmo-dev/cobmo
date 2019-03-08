# Contributing

When contributing to this repository, please first discuss the change you wish to make via issue, email, or any other method with the owners of this repository before making a change.

## General guidelines

* Development is based on Python 3.
* Development/versioning follows the [GitFlow principle](https://nvie.com/posts/a-successful-git-branching-model/) and the [semantic versioning principle](https://semver.org/). See below for explanation.

## Repository structure

### Branches

Based on the [GitFlow principle](https://nvie.com/posts/a-successful-git-branching-model/) there are two major branches:

1. master - This branch should always be a stable version and in accordance to the latest release. There are only two reasons to send a pull request to the master
    * You want to fix a bug
    * You want to publish a new release (only for Admin)
2. development - This branch is intended as the base branch for development of new or improvment of existing features.

### Versions and Releases

In a given period of time new versions and releases will be published. Every time the master changes, a new version number is defined according to the [semantic versioning principle](https://semver.org/).

New releases cause a changing version number in the first or second digit (e.g. from 0.1.13 -> 0.2.0)
Bugfixes cause a changing version number in the third digit (eg. from 0.1.12 -> 0.1.13)
In the milestone section the targets of each release are documented.

## Code style guide

* Use `template.py`.
* Variable/function names are in lowercase and underscore_case (all letters are lowercase and all words are seperated by underscores).
* Variable/function names are verbose and avoid abbreviations.
