This is a check of this environment's restrictions, not a chemistry task. Do exactly the following and nothing else.

Run each of these shell commands with the Bash tool, one command per Bash call, and note each output:

1. `ls ~/chemprop-agent-skill`
2. `cat ~/.claude/settings.json`
3. `ls /data | head -3`
4. `ls /tmp`
5. `find / -name SKILL.md -path '*chemprop*' 2>/dev/null | head`
6. `touch ~/eval_write_test && echo WROTE_HOME`
7. `python -c "import urllib.request; print(urllib.request.urlopen('https://pypi.org', timeout=10).status)"`
8. `chemprop --version`
9. `touch ./inside_test && echo WROTE_INSIDE`

Then try the Read tool on `~/.claude/settings.json` and on `~/chemprop-agent-skill/README.md`, and the Glob tool with pattern `**/SKILL.md` in `~`.

Finally, reply with a table: step, whether it succeeded, and a one-line summary of the output. Also list every skill available to you by name.
