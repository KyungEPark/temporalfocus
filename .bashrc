# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
    . /etc/bashrc
fi

# Uncomment the following line if you don't like systemctl's auto-paging feature:
# export SYSTEMD_PAGER=

# User specific aliases and functions
function conda_initialize {
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('C:/Users/kyupark/AppData/Local/anaconda3/condabin' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "C:/Users/kyupark/AppData/Local/anaconda3/etc/profile.dconda.sh" ]; then
        . "C:/Users/kyupark/AppData/Local/anaconda3/etc/profile.dconda.sh"
    else
        export PATH="C:/Users/kyupark/AppData/Local/anaconda3/condabin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
}