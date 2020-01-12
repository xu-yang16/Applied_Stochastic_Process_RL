@echo off
setlocal enableExtensions
setlocal enableDelayedExpansion
:: init
git init
:: choose github or tsinghua
echo #####choose github(1) or tsinghua(2)#####
:: input
set /p choice=
:: name of repository
echo #####input the name of your repository#####
set /p name=
if "%choice%"=="1" (
  git config user.name "xu-yang16"
  git config user.email "2306669517@qq.com"
  SET url=https://github.com/xu-yang16/%name%.git
  echo adding !url!
  git remote add origin !url!
)else (
  git config user.name "杨旭"
  git config user.email "xu-yang16@mails.tsinghua.edu.cn"
  SET url=git@git.tsinghua.edu.cn:xu-yang16/%name%.git
  echo adding !url!
  git remote add origin !url!
)
:: return
echo ################Finish!################
pause
exit
