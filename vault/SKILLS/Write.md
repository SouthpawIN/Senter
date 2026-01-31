---
description: Writes content to a file.
---
### CAPABILITY: Write File
Use this to create or overwrite a file.
<tool_code>bash(cmd="cat <<'EOFILE' > '{{file_path}}'\n{{content}}\nEOFILE")</tool_code>
