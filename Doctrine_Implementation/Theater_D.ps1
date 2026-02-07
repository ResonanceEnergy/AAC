<# 
.SYNOPSIS
    Theater D - Information Asymmetry Operations
    
.DESCRIPTION
    Implements the Information Asymmetry theater for detecting and exploiting
    information advantages and data gaps in the market.
    
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
    Name = "Theater_D"
    Description = "Information Asymmetry"
    Agents = @(
        "api_scanner",
        "data_gap_finder",
        "access_arbitrage",
        "network_mapper"
    )
    RefreshIntervalSeconds = 180  # Moderate refresh - info signals have medium decay
    ConfidenceThreshold = 0.60    # Lower threshold - info asymmetry is more speculative
    MaxConcurrentSignals = 8
    MinInformationEdge = 0.3      # Minimum perceived edge for action
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
    $logEntry = "[$timestamp] [$Level] [Theater_D] $Message"
    
    $logFile = Join-Path $LogDir "theater_d_$(Get-Date -Format 'yyyyMMdd').log"
    Add-Content -Path $logFile -Value $logEntry
    
    if ($Verbose -or $Level -eq "ERROR") {
        switch ($Level) {
            "ERROR" { Write-Host $logEntry -ForegroundColor Red }
            "WARN"  { Write-Host $logEntry -ForegroundColor Yellow }
            "INFO"  { Write-Host $logEntry -ForegroundColor Magenta }
            default { Write-Host $logEntry }
        }
    }
}

# ============================================
# TRIGGER DETECTION
# ============================================

class InfoSignal {
    [string]$SignalId
    [string]$SignalType
    [string]$Subject        # What the info is about
    [double]$EdgeEstimate   # Estimated information advantage (0-1)
    [double]$Confidence
    [string]$Source
    [datetime]$DetectedAt
    [string]$Category       # 'early_access', 'data_gap', 'network_intel', 'api_discovery'
    [hashtable]$Metadata
    
    InfoSignal([string]$type, [string]$subject, [double]$edge, [double]$confidence, [string]$category) {
        $this.SignalId = [guid]::NewGuid().ToString().Substring(0,8)
        $this.SignalType = $type
        $this.Subject = $subject
        $this.EdgeEstimate = $edge
        $this.Confidence = $confidence
        $this.Category = $category
        $this.DetectedAt = Get-Date
        $this.Metadata = @{}
    }
    
    [double] GetActionScore() {
        # Combined score for prioritization
        return $this.EdgeEstimate * $this.Confidence
    }
}

function Invoke-TriggerDetection {
    <#
    .SYNOPSIS
        Detects information-based trading triggers
    #>
    
    Write-TheaterLog "Starting information asymmetry scan..."
    
    $triggers = @()
    
    # 1. Scan APIs for early signals
    $apiSignals = Get-APISignals
    foreach ($signal in $apiSignals) {
        if ($signal.GetActionScore() -ge $script:TheaterConfig.MinInformationEdge) {
            $triggers += $signal
            Write-TheaterLog "API signal: $($signal.Subject) edge=$($signal.EdgeEstimate)" "INFO"
        }
    }
    
    # 2. Find data gaps
    $gapSignals = Get-DataGapSignals
    foreach ($signal in $gapSignals) {
        if ($signal.Confidence -ge $script:TheaterConfig.ConfidenceThreshold) {
            $triggers += $signal
            Write-TheaterLog "Data gap: $($signal.Subject)" "INFO"
        }
    }
    
    # 3. Check access arbitrage opportunities
    $accessSignals = Get-AccessSignals
    foreach ($signal in $accessSignals) {
        if ($signal.GetActionScore() -ge $script:TheaterConfig.MinInformationEdge) {
            $triggers += $signal
            Write-TheaterLog "Access opportunity: $($signal.Subject)" "INFO"
        }
    }
    
    # 4. Analyze network intelligence
    $networkSignals = Get-NetworkSignals
    foreach ($signal in $networkSignals) {
        if ($signal.Confidence -ge $script:TheaterConfig.ConfidenceThreshold) {
            $triggers += $signal
            Write-TheaterLog "Network intel: $($signal.Subject)" "INFO"
        }
    }
    
    Write-TheaterLog "Information scan complete: $($triggers.Count) triggers detected"
    
    return $triggers
}

function Get-APISignals {
    <#
    .SYNOPSIS
        Calls Python API scanner agent via wrapper
    #>
    
    $signals = @()
    
    try {
        $wrapperScript = Join-Path $PSScriptRoot ".." "shared" "powershell_agent_wrapper.py"
        
        if (Test-Path $wrapperScript) {
            $result = & python $wrapperScript "api_scanner"
            
            foreach ($line in $result) {
                if ($line.Trim() -and -not $line.StartsWith("ERROR:")) {
                    $parts = $line.Split('|')
                    if ($parts.Count -ge 3) {
                        $signal = [InfoSignal]::new(
                            "api_discovery",
                            $parts[0],
                            0.5,  # Default edge estimate
                            [double]$parts[1],
                            "api_discovery"
                        )
                        $signal.Source = "api_scanner"
                        $signals += $signal
                    }
                }
                elseif ($line.StartsWith("ERROR:")) {
                    Write-TheaterLog "API scanner error: $line" "ERROR"
                }
            }
        }
        else {
            Write-TheaterLog "Wrapper script not found: $wrapperScript" "ERROR"
        }
    }
    catch {
        Write-TheaterLog "API scanning error: $_" "ERROR"
    }
    
    return $signals
}

function Get-DataGapSignals {
    <#
    .SYNOPSIS
        Calls Python data gap finder agent via wrapper
    #>
    
    $signals = @()
    
    try {
        $wrapperScript = Join-Path $PSScriptRoot ".." "shared" "powershell_agent_wrapper.py"
        
        if (Test-Path $wrapperScript) {
            $result = & python $wrapperScript "data_gap_finder"
            
            foreach ($line in $result) {
                if ($line.Trim() -and -not $line.StartsWith("ERROR:")) {
                    $parts = $line.Split('|')
                    if ($parts.Count -ge 2) {
                        $signal = [InfoSignal]::new(
                            "data_gap",
                            $parts[0],
                            0.4,  # Data gaps have moderate edge
                            [double]$parts[1],
                            "data_gap"
                        )
                        $signal.Source = "data_gap_finder"
                        $signals += $signal
                    }
                }
                elseif ($line.StartsWith("ERROR:")) {
                    Write-TheaterLog "Data gap finder error: $line" "ERROR"
                }
            }
        }
        else {
            Write-TheaterLog "Wrapper script not found: $wrapperScript" "ERROR"
        }
    }
    catch {
        Write-TheaterLog "Data gap analysis error: $_" "ERROR"
    }
    
    return $signals
}

function Get-AccessSignals {
    <#
    .SYNOPSIS
        Calls Python access arbitrage agent via wrapper
    #>
    
    $signals = @()
    
    try {
        $wrapperScript = Join-Path $PSScriptRoot ".." "shared" "powershell_agent_wrapper.py"
        
        if (Test-Path $wrapperScript) {
            $result = & python $wrapperScript "access_arbitrage"
            
            foreach ($line in $result) {
                if ($line.Trim() -and -not $line.StartsWith("ERROR:")) {
                    $parts = $line.Split('|')
                    if ($parts.Count -ge 2) {
                        $signal = [InfoSignal]::new(
                            "access_opportunity",
                            $parts[0],
                            0.6,  # Access opportunities have higher edge
                            [double]$parts[1],
                            "early_access"
                        )
                        $signal.Source = "access_arbitrage"
                        $signals += $signal
                    }
                }
                elseif ($line.StartsWith("ERROR:")) {
                    Write-TheaterLog "Access arbitrage error: $line" "ERROR"
                }
            }
        }
        else {
            Write-TheaterLog "Wrapper script not found: $wrapperScript" "ERROR"
        }
    }
    catch {
        Write-TheaterLog "Access arbitrage error: $_" "ERROR"
    }
    
    return $signals
}

function Get-NetworkSignals {
    <#
    .SYNOPSIS
        Calls Python network mapper agent via wrapper
    #>
    
    $signals = @()
    
    try {
        $wrapperScript = Join-Path $PSScriptRoot ".." "shared" "powershell_agent_wrapper.py"
        
        if (Test-Path $wrapperScript) {
            $result = & python $wrapperScript "network_mapper"
            
            foreach ($line in $result) {
                if ($line.Trim() -and -not $line.StartsWith("ERROR:")) {
                    $parts = $line.Split('|')
                    if ($parts.Count -ge 2) {
                        $signal = [InfoSignal]::new(
                            "network_change",
                            $parts[0],
                            0.35,  # Network intel has lower immediate edge
                            [double]$parts[1],
                            "network_intel"
                        )
                        $signal.Source = "network_mapper"
                        $signals += $signal
                    }
                }
                elseif ($line.StartsWith("ERROR:")) {
                    Write-TheaterLog "Network mapper error: $line" "ERROR"
                }
            }
        }
        else {
            Write-TheaterLog "Wrapper script not found: $wrapperScript" "ERROR"
        }
    }
    catch {
        Write-TheaterLog "Network mapping error: $_" "ERROR"
    }
    
    return $signals
}

# ============================================
# ACTION EXECUTION
# ============================================

function Invoke-TheaterAction {
    param(
        [Parameter(Mandatory)]
        [InfoSignal]$Signal
    )
    
    Write-TheaterLog "Executing action for signal: $($Signal.SignalId) - $($Signal.SignalType)"
    
    if ($DryRun) {
        Write-TheaterLog "[DRY RUN] Would execute: $($Signal.SignalType) on $($Signal.Subject)" "INFO"
        return
    }
    
    # Determine action based on signal category
    switch ($Signal.Category) {
        "api_discovery" {
            # Act on API-discovered information
            $action = @{
                Type = "INFO_TRADE"
                Subject = $Signal.Subject
                EdgeEstimate = $Signal.EdgeEstimate
                Strategy = "FRONT_RUN"
                Priority = if ($Signal.EdgeEstimate -gt 0.7) { "HIGH" } else { "MEDIUM" }
                Reason = "API signal detected"
            }
        }
        "data_gap" {
            # Research and potentially act on data gap
            $action = @{
                Type = "RESEARCH"
                Subject = $Signal.Subject
                EdgeEstimate = $Signal.EdgeEstimate
                Strategy = "DEEP_DIVE"
                Priority = "LOW"
                Reason = "Data gap identified"
            }
        }
        "early_access" {
            # Capitalize on early access
            $action = @{
                Type = "EARLY_POSITION"
                Subject = $Signal.Subject
                EdgeEstimate = $Signal.EdgeEstimate
                Strategy = "ACCUMULATE"
                Priority = "HIGH"
                Reason = "Early access opportunity"
            }
        }
        "network_intel" {
            # Monitor and position based on network intel
            $action = @{
                Type = "MONITOR"
                Subject = $Signal.Subject
                EdgeEstimate = $Signal.EdgeEstimate
                Strategy = "WATCH_AND_WAIT"
                Priority = "LOW"
                Reason = "Network intelligence"
            }
        }
        default {
            Write-TheaterLog "Unknown signal category: $($Signal.Category)" "WARN"
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
            Subject = $Signal.Subject
            Category = $Signal.Category
            EdgeEstimate = $Signal.EdgeEstimate
            Confidence = $Signal.Confidence
            ActionScore = $Signal.GetActionScore()
        }
        Action = $action
    }
    
    $existingActions = @()
    if (Test-Path $actionLog) {
        $existingActions = Get-Content $actionLog | ConvertFrom-Json
    }
    $existingActions += $actionEntry
    $existingActions | ConvertTo-Json -Depth 10 | Set-Content $actionLog
    
    Write-TheaterLog "Action logged: $($action.Type) - $($action.Strategy) on $($Signal.Subject)"
}

# ============================================
# MAIN LOOP
# ============================================

function Start-TheaterD {
    param(
        [switch]$Continuous
    )
    
    Write-TheaterLog "=== Theater D Starting ===" "INFO"
    Write-TheaterLog "Agents: $($script:TheaterConfig.Agents -join ', ')"
    Write-TheaterLog "Confidence Threshold: $($script:TheaterConfig.ConfidenceThreshold)"
    Write-TheaterLog "Min Information Edge: $($script:TheaterConfig.MinInformationEdge)"
    
    do {
        try {
            # Detect triggers
            $triggers = Invoke-TriggerDetection
            
            # Sort by action score (edge * confidence)
            $sortedTriggers = $triggers | Sort-Object -Property { $_.GetActionScore() } -Descending
            
            # Process triggers (limited to max concurrent)
            $processCount = [Math]::Min($sortedTriggers.Count, $script:TheaterConfig.MaxConcurrentSignals)
            
            for ($i = 0; $i -lt $processCount; $i++) {
                Invoke-TheaterAction -Signal $sortedTriggers[$i]
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
    
    Write-TheaterLog "=== Theater D Stopped ===" "INFO"
}

# Entry point
if ($MyInvocation.InvocationName -ne '.') {
    Start-TheaterD
}
