
#!/bin/sh

cd ~/hw5

# echo *.png > .gitignore
# echo ./samples/*.png > .gitignore
# test syncrepo.sh

git add *
git commit -m "${1}"
git push -u origin master