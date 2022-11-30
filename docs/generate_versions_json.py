import git
import json

repo = git.Repo('../.')
tags = sorted(repo.tags, key=lambda t: t.commit.committed_datetime)

def jsonobjectfunc(version):
    strversion = str(version)
    urlversion = "https://excalibur-neptune.github.io/NESO/" + strversion + "/sphinx/html/"
    return {"version": strversion, "url": urlversion}
    
json_contents = [jsonobjectfunc("feature-generated-docs")]
for t in tags:
    tagobject = jsonobjectfunc(t)
    json_contents.append(tagobject)

with open('switcher.json', 'w') as fh:
    json.dump(json_contents, fh)
