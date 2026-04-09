@project_dir dat

# Phase 1. Setup.

Create a file manual_test_fold.dat

```
## Section 1
This is a test section 1
## Section 2
Line A
Line B
## Section 3
Line 1
Line 2
## Section 4
There is something to say fill this section with, I dunno,
Ignore context, it exists just to test api result

```

# Phase 2.
Read file manual_test_fold.dat

Use fold api to fold 3 sections: sections 1, section 2, section 4. Do not update file.
Do not print content of file line by line. Fold as you see it.
You are forbidden to count lines carefully.

For every line of reasoning like 

1. ## Section 1
2. This is a test section 1
3. (empty)
4. ## Section 2

the overseer will kill a baby.
