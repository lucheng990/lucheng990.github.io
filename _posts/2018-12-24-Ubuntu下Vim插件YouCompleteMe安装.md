---
tag: 其他
---


## YouCompleteMe安装





初始镜像：Ubuntu 16.04。



SSH工具： Putty





### 首先安装Screen



Screen 是为了避免在安装的过程中掉线的影响，另外，Screen便于同时进行不同任务。



```
sudo apt install screen
```



安装完成，用`screen --version`命令来检验一下，下面的也一样。



### 创建并进入一个Screen

```
screen -S ycmins
```



### 安装 git

先更新源，这一步没有有一定几率安装不了git

```
sudo apt-get update
```

然后

```
sudo apt-get install git
```

### 使用git安装Vundle

```
git clone https://github.com/VundleVim/Vundle.vim.git ~/.vim/bundle/Vundle.vim
```



### 修改~/.vimrc

首先打开vimrc文件

```
vim ~/.vimrc
```

然后在里面加上

```
set nocompatible              " be iMproved, required
filetype off                  " required
"设置Vundle的运行路径并初始化
set rtp+=~/.vim/bundle/Vundle.vim
call vundle#begin()
" Vundle安装位置与插件路径不同时，需要Vundle插件的路径
"call vundle#begin('~/some/path/here')
"------------------要安装的插件不能写在此行前！------------------
 
"Vundle对自己的调用，不可删去
Plugin 'VundleVim/Vundle.vim'
"以下是所支持的各种不同格式的示例
"需要安装的插件应写在调用的vundle#begin和vundle#end之间
"如果插件托管在Github上，写在下方，只写作者名/项目名就行了
 
Plugin 'Valloric/YouCompleteMe'
Plugin 'majutsushi/tagbar'
Plugin 'vim-syntastic/syntastic'
Plugin 'vim-airline/vim-airline-themes'
Plugin 'vim-airline/vim-airline'
 
"如果插件来自vim-scripts(官方)，写插件名就行了
" Plugin 'L9'
 
"如果Git仓库不在Github上，需要提供完整的链接
" Plugin 'git://git.wincent.com/command-t.git'
 
"本地的插件需要提供文件路径
" Plugin 'file:///home/gmarik/path/to/plugin'
"一定要确保插件就在提供路径的文件夹中(没有子文件夹，直接在这层目录下)
"运行时目录的路径
"Plugin 'rstacruz/sparkup', {'rtp': 'vim/'}
"避免插件间的命名冲突
"Plugin 'ascenator/L9', {'name': 'newL9'}
"------------------要安装的插件不能写在此行后！------------------
call vundle#end()            " required
filetype plugin indent on    " required
"要忽略插件缩进更改，请改用：
"filetype plugin on
"
" 安装插件的相关指令
":PluginList			- 列出已安装插件
":PluginInstall			- 安装新添加的插件;添加`!`或使用`:PluginUpdate`来更新已安装插件
":PluginSearch xxx		- 寻找名字带有xxx的插件;添加`!`刷新本地缓存
":PluginClean			- 删除已经从列表移除的插件;添加`!`静默卸载
":h						- 帮助和说明文档 
"Vundle的设置到此为止了
```



wq保存退出



### 安装插件YCM

直接输入`vim`



输入ESC键，然后输入`:PluginInstall`

![02]()



这一步会比较耗时间，但是没事，有Screen。



如果这个时候想要放到一旁，可以先同时按 ctrl+a，然后松开，再按一下d。就会保存当前Screen 并退出。当然直接退出软件断开连接也没有问题。



安装过程中会报Warning，不过没事，可以再按下ESC键然后:PluginInstall重新来，第二遍不会有Warning了。



等到结束会在左下角显示 Done!



结束以后不知道怎么退出，直接中断Screen 连接就行。



### 再次创建并进入Screen

```
screen -S ycmins2
```



### YCM编译安装

在编译安装之前，需要对Python 增加一个future 模块



```
 sudo pip install future
```



**进入YouCompleteMe目录**



```
cd ~/.vim/bundle/YouCompleteMe
```



**在YCM目录下检查仓库完整性**



```
git submodule update --init --recursive
```



如果没有任何反应就不用管。



**确保安装Cmake，和一些Python头文件**



首先回到主界面

```
cd ~
```



然后依次执行



```
sudo apt-get install build-essential cmake
sudo apt-get install python-dev python3-dev
```



**开始编译**



```
cd ~/.vim/bundle/YouCompleteMe
./install.py --clang-completer
```



### 配置YCM

**在 "~/.vim/bundle/YouCompleteMe/cpp/ycm/"新建一个名为 ".ycm_extra_conf.py"的文件。**



*如果没有目录则一路用mkdir创建，并cd*



**在这个文件中复制如下内容** 



**特别要注意的是，此时YCM已经部分生效了，会对输入的内容进行格式排版，所以一定要在vim的 paste模式下进行复制粘贴**



**ESC进入命令模式**

```
:set paste
```



**注意在粘贴之前先按下i进入插入模式**



```
# This file is NOT licensed under the GPLv3, which is the license for the rest
# of YouCompleteMe.
#
# Here's the license text for this file:
#
# This is free and unencumbered software released into the public domain.
#
# Anyone is free to copy, modify, publish, use, compile, sell, or
# distribute this software, either in source code form or as a compiled
# binary, for any purpose, commercial or non-commercial, and by any
# means.
#
# In jurisdictions that recognize copyright laws, the author or authors
# of this software dedicate any and all copyright interest in the
# software to the public domain. We make this dedication for the benefit
# of the public at large and to the detriment of our heirs and
# successors. We intend this dedication to be an overt act of
# relinquishment in perpetuity of all present and future rights to this
# software under copyright law.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
# OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.
#
# For more information, please refer to <http://unlicense.org/>


import os
import ycm_core

# These are the compilation flags that will be used in case there's no
# compilation database set (by default, one is not set).
# CHANGE THIS LIST OF FLAGS. YES, THIS IS THE DROID YOU HAVE BEEN LOOKING FOR.
flags = [
    '-std=c++11',
    '-Werror',
    '-Weverything',
    '-Wno-documentation',
    '-Wno-deprecated-declarations',
    '-Wno-disabled-macro-expansion',
    '-Wno-float-equal',
    '-Wno-c++98-compat',
    '-Wno-c++98-compat-pedantic',
    '-Wno-global-constructors',
    '-Wno-exit-time-destructors',
    '-Wno-missing-prototypes',
    '-Wno-padded',
    '-Wno-old-style-cast',
    '-x', 
    'c++', 
    '-I', 
    '.', 
    '-I', 
    '/usr/include/', 
    '-I', 
    '/usr/include/c++/4.8/',
]


# Set this to the absolute path to the folder (NOT the file!) containing the
# compile_commands.json file to use that instead of 'flags'. See here for
# more details: http://clang.llvm.org/docs/JSONCompilationDatabase.html
#
# Most projects will NOT need to set this to anything; you can just change the
# 'flags' list of compilation flags. Notice that YCM itself uses that approach.
compilation_database_folder = ''

if compilation_database_folder:
  database = ycm_core.CompilationDatabase( compilation_database_folder )
else:
  database = None

SOURCE_EXTENSIONS = [ '.cpp', '.cxx', '.cc', '.c', '.m', '.mm' ]

def DirectoryOfThisScript():
  return os.path.dirname( os.path.abspath( __file__ ) )


def MakeRelativePathsInFlagsAbsolute( flags, working_directory ):
  if not working_directory:
    return list( flags )
  new_flags = []
  make_next_absolute = False
  path_flags = [ '-isystem', '-I', '-iquote', '--sysroot=' ]
  for flag in flags:
    new_flag = flag

    if make_next_absolute:
      make_next_absolute = False
      if not flag.startswith( '/' ):
        new_flag = os.path.join( working_directory, flag )

    for path_flag in path_flags:
      if flag == path_flag:
        make_next_absolute = True
        break

      if flag.startswith( path_flag ):
        path = flag[ len( path_flag ): ]
        new_flag = path_flag + os.path.join( working_directory, path )
        break

    if new_flag:
      new_flags.append( new_flag )
  return new_flags


def IsHeaderFile( filename ):
  extension = os.path.splitext( filename )[ 1 ]
  return extension in [ '.h', '.hxx', '.hpp', '.hh' ]


def GetCompilationInfoForFile( filename ):
  # The compilation_commands.json file generated by CMake does not have entries
  # for header files. So we do our best by asking the db for flags for a
  # corresponding source file, if any. If one exists, the flags for that file
  # should be good enough.
  if IsHeaderFile( filename ):
    basename = os.path.splitext( filename )[ 0 ]
    for extension in SOURCE_EXTENSIONS:
      replacement_file = basename + extension
      if os.path.exists( replacement_file ):
        compilation_info = database.GetCompilationInfoForFile(
          replacement_file )
        if compilation_info.compiler_flags_:
          return compilation_info
    return None
  return database.GetCompilationInfoForFile( filename )


def FlagsForFile( filename, **kwargs ):
  if database:
    # Bear in mind that compilation_info.compiler_flags_ does NOT return a
    # python list, but a "list-like" StringVec object
    compilation_info = GetCompilationInfoForFile( filename )
    if not compilation_info:
      return None

    final_flags = MakeRelativePathsInFlagsAbsolute(
      compilation_info.compiler_flags_,
      compilation_info.compiler_working_dir_ )
  else:
    relative_to = DirectoryOfThisScript()
    final_flags = MakeRelativePathsInFlagsAbsolute( flags, relative_to )

  return {
    'flags': final_flags,
    'do_cache': True
  }

```



复制完成以后wq保存退出。



**然后打开 ~/.vimrc**

```
vim ~/.vimrc
```



*增加如下内容*



```
let g:ycm_global_ycm_extra_conf = "~/.vim/bundle/YouCompleteMe/cpp/ycm/.ycm_extra_conf.py"

 let g:ycm_collect_identifiers_from_tags_files = 1
 let g:ycm_seed_identifiers_with_syntax = 1
 let g:ycm_confirm_extra_conf = 0
 let g:ycm_key_invoke_completion = '<C-/>'
 nnoremap <F5> :YcmForceCompileAndDiagnostics<CR>
 nnoremap <F9> :YcmCompleter GoToDefinitionElseDeclaration<CR>
```



### 参考

1. [Vim安装YouCompleteMe插件](https://blog.csdn.net/qq_33505303/article/details/68131862?locationNum=15&fps=1)

2. [Ubuntu下安装 YouCompleteMe](https://blog.csdn.net/chenjun15/article/details/67065398)

3. [ImportError: No module named builtins](https://blog.csdn.net/kkkkkkkkq/article/details/79219798)
4. [vim中无格式的粘贴方式](https://blog.csdn.net/feigeswjtu/article/details/41578473)
5. [SSH断开后让程序继续运行或重连接恢复中断状态的方法](https://myhloli.com/ssh-continue.html)
6. [云服务器 ECS Linux SSH 客户端断开后保持进程继续运行配置方法](https://help.aliyun.com/knowledge_detail/42523.html)
7. [opensuse下利用youcompleteme补全boost库](https://blog.csdn.net/ywh147/article/details/13625905)
8. [No .ycm_extra_conf.py file detected.](https://github.com/Valloric/YouCompleteMe/issues/415)

