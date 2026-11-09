Push a specific file to GitHub with a commit message.

Usage: /push-to-git <filename> "<commit message>"

Arguments: $ARGUMENTS

Parse $ARGUMENTS as follows:
- The first whitespace-delimited token is the filename
- Everything after the first token (stripping surrounding quotes if present) is the commit message

Then run these git commands in sequence, stopping and reporting an error if any step fails:

1. `git add <filename>`
2. `git commit -m "<commit message>"` — use a HEREDOC to pass the message so quotes and special characters are handled correctly
3. `git push`

Report the output of each step clearly. If the push succeeds, confirm the filename and commit message that were used.
