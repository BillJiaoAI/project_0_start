$user32 = Add-Type -MemberDefinition @"
[DllImport("user32.dll")]
public static extern void keybd_event(byte bVk, byte bScan, uint dwFlags, UIntPtr dwExtraInfo);
[DllImport("user32.dll")]
public static extern short GetAsyncKeyState(int vKey);
"@ -Name "User32Utils" -Namespace "Win32" -PassThru

$VK_3 = 0x33
$VK_5 = 0x35

function Press-Key {
    param([byte]$keyCode)
    $user32::keybd_event($keyCode, 0, 0, [UIntPtr]::Zero)
    Start-Sleep -Milliseconds 50
    $user32::keybd_event($keyCode, 0, 2, [UIntPtr]::Zero)
}

function Is-Key-Pressed {
    param([byte]$keyCode)
    return ($user32::GetAsyncKeyState($keyCode) -band 0x8000) -ne 0
}

Write-Host "自动按3脚本启动..."
Write-Host "每0.5s-1s随机按下3键"
Write-Host "按下5键停止脚本"

while ($true) {
    if (Is-Key-Pressed $VK_5) {
        Write-Host "检测到5键，停止脚本"
        break
    }
    
    Press-Key $VK_3
    $delay = (Get-Random -Minimum 500 -Maximum 1000) / 1000.0
    Write-Host "已按下3键，下次按键在 $($delay.ToString("F2"))s 后"
    
    $startTime = Get-Date
    while (((Get-Date) - $startTime).TotalSeconds -lt $delay) {
        if (Is-Key-Pressed $VK_5) {
            Write-Host "检测到5键，停止脚本"
            return
        }
        Start-Sleep -Milliseconds 10
    }
}