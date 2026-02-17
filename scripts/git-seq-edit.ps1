param($todoFile)
# Replace the pick for the offending commit with edit
(Get-Content -Raw $todoFile) -replace 'pick e83d932', 'edit e83d932' | Set-Content $todoFile
