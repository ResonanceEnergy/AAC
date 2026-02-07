<# 
.SYNOPSIS
    Theater B - Attention Arbitrage Operations
    
.DESCRIPTION
    Implements the Attention/Narrative theater for detecting and exploiting
    attention-based market inefficiencies.
    
.NOTES
    Part of ACC Doctrine Implementation
    Version: 1.0.0
#>

param(
    [switch]$Verbose,
    [switch]$DryRun,
    [string]$ConfigPath = "$PSScriptRoot\Doctrine_Config.yaml"
)

$ErrorActionPreference = "Stop"

# ============================================
# CONFIGURATION
# ============================================

$script:TheaterConfig = @{
    Name = "Theater_B"
    Description = "Attention Arbitrage"
    Agents = @(
        "narrative_analyzer",
        "engagement_predictor", 
        "content_optimizer"
    )
    RefreshIntervalSeconds = 300
    ConfidenceThreshold = 0.65
    MaxConcurrentSignals = 10
}

# ============================================
# LOGGING
# ============================================

$LogDir = Join-Path $PSScriptRoot "logs"
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

function Write-TheaterLog {
    param([string]$Message, [string]$Level = "INFO")
    
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logEntry = "[$timestamp] [$Level] [Theater_B] $Message"
    
    $logFile = Join-Path $LogDir "theater_b_$(Get-Date -Format 'yyyyMMdd').log"
    Add-Content -Path $logFile -Value $logEntry
    
    if ($Verbose -or $Level -eq "ERROR") {
        switch ($Level) {
            "ERROR" { Write-Host $logEntry -ForegroundColor Red }
            "WARN"  { Write-Host $logEntry -ForegroundColor Yellow }
            "INFO"  { Write-Host $logEntry -ForegroundColor Cyan }
            default { Write-Host $logEntry }
        }
    }
}

# ============================================
# TRIGGER DETECTION
# ============================================

class AttentionSignal {
    [string]$SignalId
    [string]$SignalType
    [string]$Asset
    [double]$Intensity
    [double]$Confidence
    [string]$Source
    [datetime]$DetectedAt
    [hashtable]$Metadata
    
    AttentionSignal([string]$type, [string]$asset, [double]$intensity, [double]$confidence) {
        $this.SignalId = [guid]::NewGuid().ToString().Substring(0,8)
        $this.SignalType = $type
        $this.Asset = $asset
        $this.Intensity = $intensity
        $this.Confidence = $confidence
        $this.DetectedAt = Get-Date
        $this.Metadata = @{}
    }
}

function Invoke-TriggerDetection {
    <#
    .SYNOPSIS
        Detects attention-based trading triggers
    #>
    
    Write-TheaterLog "Starting trigger detection scan..."
    
    $triggers = @()
    
    # 1. Check for narrative shifts
    $narrativeSignals = Get-NarrativeSignals
    foreach ($signal in $narrativeSignals) {
        if ($signal.Confidence -ge $script:TheaterConfig.ConfidenceThreshold) {
            $triggers += $signal
            Write-TheaterLog "Narrative trigger: $($signal.Asset) - $($signal.SignalType)" "INFO"
        }
    }
    
    # 2. Check for engagement spikes
    $engagementSignals = Get-EngagementSignals
    foreach ($signal in $engagementSignals) {
        if ($signal.Intensity -gt 2.0 -and $signal.Confidence -ge $script:TheaterConfig.ConfidenceThreshold) {
            $triggers += $signal
            Write-TheaterLog "Engagement trigger: $($signal.Asset) intensity=$($signal.Intensity)" "INFO"
        }
    }
    
    # 3. Check for content momentum
    $contentSignals = Get-ContentMomentumSignals
    foreach ($signal in $contentSignals) {
        if ($signal.Confidence -ge $script:TheaterConfig.ConfidenceThreshold) {
            $triggers += $signal
        }
    }
    
    Write-TheaterLog "Trigger scan complete: $($triggers.Count) triggers detected"
    
    return $triggers
}

function Get-NarrativeSignals {
    <#
    .SYNOPSIS
        Calls Python narrative analyzer agent
    #>
    
    $signals = @()
    
    try {
        $pythonScript = Join-Path $PSScriptRoot ".." "BigBrainIntelligence" "agents.py"
        
        if (Test-Path $pythonScript) {
            # Run the agent and capture output
            $result = python -c @"
import sys
sys.path.insert(0, r'$((Join-Path $PSScriptRoot "..").Replace('\', '\\'))')
import asyncio
from BigBrainIntelligence.agents import get_agent
agent = get_agent('narrative_analyzer')
if agent:
    findings = asyncio.run(agent.run_scan())
    for f in findings:
        print(f'{f.data.get("narrative", "unknown")}|{f.confidence}|{f.urgency}')
"@
            
            foreach ($line in $result -split "`n") {
                if ($line.Trim()) {
                    $parts = $line.Split('|')
                    if ($parts.Count -ge 3) {
                        $signal = [AttentionSignal]::new(
                            "narrative_shift",
                            $parts[0],
                            1.0,
                            [double]$parts[1]
                        )
                        $signal.Source = "narrative_analyzer"
                        $signals += $signal
                    }
                }
            }
        }
    }
    catch {
        Write-TheaterLog "Narrative analysis error: $_" "ERROR"
    }
    
    return $signals
}

function Get-EngagementSignals {
    <#
    .SYNOPSIS
        Calls Python engagement predictor agent
    #>
    
    $signals = @()
    
    try {
        $pythonScript = Join-Path $PSScriptRoot ".." "BigBrainIntelligence" "agents.py"
        
        if (Test-Path $pythonScript) {
            $result = python -c @"
import sys
sys.path.insert(0, r'$((Join-Path $PSScriptRoot "..").Replace('\', '\\'))')
import asyncio
from BigBrainIntelligence.agents import get_agent
agent = get_agent('engagement_predictor')
if agent:
    findings = asyncio.run(agent.run_scan())
    for f in findings:
        print(f'{f.data.get("asset", "unknown")}|{f.data.get("magnitude", 1.0)}|{f.confidence}')
"@
            
            foreach ($line in $result -split "`n") {
                if ($line.Trim()) {
                    $parts = $line.Split('|')
                    if ($parts.Count -ge 3) {
                        $signal = [AttentionSignal]::new(
                            "engagement_spike",
                            $parts[0],
                            [double]$parts[1],
                            [double]$parts[2]
                        )
                        $signal.Source = "engagement_predictor"
                        $signals += $signal
                    }
                }
            }
        }
    }
    catch {
        Write-TheaterLog "Engagement analysis error: $_" "ERROR"
    }
    
    return $signals
}

function Get-ContentMomentumSignals {
    <#
    .SYNOPSIS
        Calls Python content optimizer agent
    #>
    
    $signals = @()
    
    try {
        $pythonScript = Join-Path $PSScriptRoot ".." "BigBrainIntelligence" "agents.py"
        
        if (Test-Path $pythonScript) {
            $result = python -c @"
import sys
sys.path.insert(0, r'$((Join-Path $PSScriptRoot "..").Replace('\', '\\'))')
import asyncio
from BigBrainIntelligence.agents import get_agent
agent = get_agent('content_optimizer')
if agent:
    findings = asyncio.run(agent.run_scan())
    for f in findings:
        print(f'{f.title}|{f.confidence}')
"@
            
            foreach ($line in $result -split "`n") {
                if ($line.Trim()) {
                    $parts = $line.Split('|')
                    if ($parts.Count -ge 2) {
                        $signal = [AttentionSignal]::new(
                            "content_momentum",
                            $parts[0],
                            1.0,
                            [double]$parts[1]
                        )
                        $signal.Source = "content_optimizer"
                        $signals += $signal
                    }
                }
            }
        }
    }
    catch {
        Write-TheaterLog "Content analysis error: $_" "ERROR"
    }
    
    return $signals
}

# ============================================
# ACTION EXECUTION
# ============================================

function Invoke-TheaterAction {
    param(
        [Parameter(Mandatory)]
        [AttentionSignal]$Signal
    )
    
    Write-TheaterLog "Executing action for signal: $($Signal.SignalId) - $($Signal.SignalType)"
    
    if ($DryRun) {
        Write-TheaterLog "[DRY RUN] Would execute: $($Signal.SignalType) on $($Signal.Asset)" "INFO"
        return
    }
    
    # Determine action based on signal type
    switch ($Signal.SignalType) {
        "narrative_shift" {
            # Position for narrative momentum
            $action = @{
                Type = "POSITION"
                Asset = $Signal.Asset
                Direction = if ($Signal.Intensity -gt 0) { "LONG" } else { "SHORT" }
                Size = "SMALL"
                Reason = "Narrative shift detected"
            }
        }
        "engagement_spike" {
            # Quick entry on engagement
            $action = @{
                Type = "SCALP"
                Asset = $Signal.Asset
                Direction = "LONG"
                Duration = "SHORT_TERM"
                Reason = "Engagement spike - intensity $($Signal.Intensity)"
            }
        }
        "content_momentum" {
            # Monitor for entry
            $action = @{
                Type = "ALERT"
                Asset = $Signal.Asset
                Reason = "Content momentum building"
            }
        }
        default {
            Write-TheaterLog "Unknown signal type: $($Signal.SignalType)" "WARN"
            return
        }
    }
    
    # Log the action
    $actionLog = Join-Path $LogDir "actions_$(Get-Date -Format 'yyyyMMdd').json"
    $actionEntry = @{
        Timestamp = (Get-Date).ToString("o")
        Signal = @{
            Id = $Signal.SignalId
            Type = $Signal.SignalType
            Asset = $Signal.Asset
            Confidence = $Signal.Confidence
        }
        Action = $action
    }
    
    $existingActions = @()
    if (Test-Path $actionLog) {
        $existingActions = Get-Content $actionLog | ConvertFrom-Json
    }
    $existingActions += $actionEntry
    $existingActions | ConvertTo-Json -Depth 10 | Set-Content $actionLog
    
    Write-TheaterLog "Action logged: $($action.Type) on $($Signal.Asset)"
}

# ============================================
# MAIN LOOP
# ============================================

function Start-TheaterB {
    param(
        [switch]$Continuous
    )
    
    Write-TheaterLog "=== Theater B Starting ===" "INFO"
    Write-TheaterLog "Agents: $($script:TheaterConfig.Agents -join ', ')"
    Write-TheaterLog "Confidence Threshold: $($script:TheaterConfig.ConfidenceThreshold)"
    
    do {
        try {
            # Detect triggers
            $triggers = Invoke-TriggerDetection
            
            # Process triggers (limited to max concurrent)
            $processCount = [Math]::Min($triggers.Count, $script:TheaterConfig.MaxConcurrentSignals)
            
            for ($i = 0; $i -lt $processCount; $i++) {
                Invoke-TheaterAction -Signal $triggers[$i]
            }
            
            if ($Continuous) {
                Write-TheaterLog "Sleeping for $($script:TheaterConfig.RefreshIntervalSeconds)s..."
                Start-Sleep -Seconds $script:TheaterConfig.RefreshIntervalSeconds
            }
        }
        catch {
            Write-TheaterLog "Theater loop error: $_" "ERROR"
            if ($Continuous) {
                Start-Sleep -Seconds 60
            }
        }
    } while ($Continuous)
    
    Write-TheaterLog "=== Theater B Stopped ===" "INFO"
}

# Entry point
if ($MyInvocation.InvocationName -ne '.') {
    Start-TheaterB
}
