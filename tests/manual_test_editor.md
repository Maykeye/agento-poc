@project_dir ./dat/

Stages to perform:

0) If `editor_test.txt` exists, delete it.
1) Enter edit mode. Using write_file there create a new sample file `editor_test.txt` with content
```
LINE01
LINE02
LINE03
LINE04
```

2) Replace `LINE02` with `line02` using search/replace
3) Replace `LINE03` with `line03` using sed
4) Print the buffer
