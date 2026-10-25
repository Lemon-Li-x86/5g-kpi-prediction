$ErrorActionPreference = 'Stop'
$reviewRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$reviewOutput = Join-Path $reviewRoot 'deliverables/review_20261025'
$pdfOutput = Join-Path $reviewOutput 'pdf'
New-Item -ItemType Directory -Path $pdfOutput -Force | Out-Null
$wordReview = New-Object -ComObject Word.Application
$wordReview.Visible = $false
$wordReview.DisplayAlerts = 0
$wordReview.AutomationSecurity = 3
try {
    foreach ($reviewFile in Get-ChildItem -LiteralPath $reviewOutput -Filter '*.docx') {
        $reviewDoc = $null
        try {
            $reviewDoc = $wordReview.Documents.Open($reviewFile.FullName, $false, $true)
            $reviewDoc.Repaginate()
            $reviewPdf = Join-Path $pdfOutput ($reviewFile.BaseName + '.pdf')
            $reviewDoc.ExportAsFixedFormat($reviewPdf, 17)
            Write-Output ($reviewFile.Name + ' pages=' + $reviewDoc.ComputeStatistics(2))
        } finally {
            if ($null -ne $reviewDoc) { $reviewDoc.Close(0) }
        }
    }
} finally {
    $wordReview.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($wordReview) | Out-Null
}
