* Bump version in `log_wmse_audio_quality/__init__.py`
* `pytest`
* Update CHANGELOG.md
* Commit and push the change with a commit message like this: "Release vx.y.z" (replace x.y.z with the package version)
* Remove any old files inside the dist folder
* `python setup.py sdist bdist_wheel`
* `python -m twine upload dist/*`
* Add a tag with name "vx.y.z" to the commit
* Go to https://github.com/nomonosound/log-wmse-audio-quality/releases and create a release where you choose the new tag
