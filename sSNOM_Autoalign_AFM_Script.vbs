'----------------------------------------------------------  
' Script: Automated Scanning & Saving with Stop File  
'----------------------------------------------------------  
Option Explicit

' Create application object
Dim objApp : Set objApp = SPM.Application
objApp.PrintStatusMsg "Script Start"
call Main()
objApp.PrintStatusMsg "Script Done"
Set objApp = Nothing

Sub Main()
	Do While objApp.IsStartingUp : Loop

	' Access necessary objects
	Dim objScan : Set objScan = objApp.Scan
	Dim objAppr : Set objAppr = objApp.Approach

	' Prompt user for folder to save CSV files
	Dim saveFolder
	saveFolder = InputBox("Enter the folder path to save scan data:", "Save Folder", "Y:\ScanData\000")
	If saveFolder = "" Then
		MsgBox "No folder selected. Script will exit."
		Exit Sub
	End If

	' Ensure folder path ends with backslash
	If Right(saveFolder, 1) <> "\" Then
		saveFolder = saveFolder & "\"
	End If
	
	' Initialize file counter
	Dim fileCounter
	fileCounter = 0

	' Create a stop file
	Dim stopFilePath
	stopFilePath = saveFolder & "stop_flag.txt"

	' Ensure stop file exists
	Dim objFS_s, objFile
	Set objFS_s = CreateObject("Scripting.FileSystemObject")
	If Not objFS_s.FileExists(stopFilePath) Then
		Set objFile = objFS_s.CreateTextFile(stopFilePath, True)
		objFile.Write ""
		objFile.Close
		Set objFile = Nothing
	End If
	objApp.PrintStatusMsg "Stop File Created"

	' Turn off approach done dialog
	objAppr.ShowApproachDoneDialog = False

	' Flag to indicate when to exit both loops
	Dim exitLoop
	exitLoop = False 

	' Automated scanning loop
	Do
		' Check if stop file has been deleted before approaching
		If Not objFS_s.FileExists(stopFilePath) Then Exit Do
		
		' Approach the tip to the surface
		objAppr.AutoStartImaging = False
		objApp.PrintStatusMsg "Reset the tip z position"
		objAppr.ApproachPos = 500e-9 '[m] Reset the tip z position
		Do While objAppr.IsMoving : Loop
		objApp.PrintStatusMsg "Start Approach"
		objAppr.StartApproach
		Do While objAppr.IsMoving : Loop
		objAppr.StartApproach
		Do While objAppr.IsMoving : Loop
		If objAppr.Status <> 3 Then
			objAppr.StartWithdraw
			MsgBox "Approach failed. Error " & objAppr.Status & ". Script exiting."
			Exit Sub
		End If
		
		' Start a single scan
		objApp.PrintStatusMsg "Start Scan"
		objScan.StartFrameUp
		Do While objScan.IsScanning 
			' Check if stop file has been deleted while scanning
			If Not objFS_s.FileExists(stopFilePath) Then 
				' Stop scanning and withdraw tip
				objScan.Stop
				objApp.PrintStatusMsg "Stop file deleted. Withdraw"
				objAppr.StartWithdraw
				Do While objAppr.IsMoving : Loop
				
				' Set flag to exit both loops
				exitLoop = True
				Exit Do
			End If
		Loop
		
		' Exit the outer loop if flag is set
		If exitLoop Then Exit Do 
		
		' Withdraw tip after scanning
		objApp.PrintStatusMsg "Scan complete. Withdrawing..."
		objAppr.StartWithdraw
		Do While objAppr.IsMoving : Loop
		objApp.PrintStatusMsg "Withdraw complete. Saving data..."
		
		' Generate timestamp for filenames
		Dim timestamp
		timestamp = Replace(Replace(Replace(Now, ":", "-"), "/", "-"), " ", "_")
		
		' Retrieve scan parameters
        Dim imgWidth, imgHeight, numPoints, numLines, centerX, centerY
        imgWidth = objScan.ImageWidth
        imgHeight = objScan.ImageHeight
        numPoints = objScan.Points
        numLines = objScan.Lines
        centerX = objScan.CenterPosX
        centerY = objScan.CenterPosY
        
        ' Calculate step sizes
        Dim xStep, yStep
        xStep = imgWidth / (numPoints - 1)
        yStep = imgHeight / (numLines - 1)

        ' Retrieve scan data
        Dim scanline, point
        Dim scanData
        scanData = ""

        For scanline = 0 To numLines - 1
            Dim yPos
            yPos = centerY - (imgHeight / 2) + (scanline * yStep)
            
            ' Get full scan line data
            Dim topoF, topoB, amplF, amplB
            topoF  = Split(objScan.GetLine(0, 1, scanline, 0, 1), ";") ' Forward, Topology
            topoB  = Split(objScan.GetLine(1, 1, scanline, 0, 1), ";") ' Backward, Topology
            amplF  = Split(objScan.GetLine(0, 2, scanline, 0, 1), ";") ' Forward, Amplitude
            amplB  = Split(objScan.GetLine(1, 2, scanline, 0, 1), ";") ' Backward, Amplitude
            
			' Append each point with X, Y, and scan data
            For point = 0 To numPoints - 1
                Dim xPos
                xPos = centerX - (imgWidth / 2) + (point * xStep)
                scanData = scanData & xPos & "," & yPos & "," & topoF(point) & "," & topoB(point) & "," & amplF(point) & "," & amplB(point) & vbCrLf
            Next
        Next

        ' Save scan data
        Dim filename, refData
        filename = saveFolder & "scan_" & Right("00" & fileCounter, 3) & "_" & timestamp & ".csv"
		refData = objScan.ImageWidth & "," & objScan.ImageHeight & "," & objScan.Scantime & "," & objScan.Points & "," & objScan.Lines & "," & objScan.Rotation & vbCrLf
		Call SaveToCSV(filename, refData, scanData)
		objApp.PrintStatusMsg "Scan data saved: " & filename
		
		' Increment file counter for next scan
		fileCounter = fileCounter + 1		
		
		' Create a Auto Align flag file
		Dim alignFilePath
		alignFilePath = saveFolder & "align_flag.txt"
		
		' Write the new topo data filename to the align file
		Dim objFS_f : Set objFS_f = CreateObject("Scripting.FileSystemObject")
		Set objFile = objFS_f.CreateTextFile(alignFilePath, True)
		objFile.Write filename
		objFile.Close
		Set objFile = Nothing
		objApp.PrintStatusMsg "Align File Created"
		
		' Check if the align file has been deleted
		Do While objFS_s.FileExists(alignFilePath) : Loop
		
	Loop

	' Reset settings before exiting
	objAppr.AutoStartImaging = True
	objAppr.ShowApproachDoneDialog = True
	
	' Cleanup
	Set objFS_s = Nothing
	Set objFS_f = Nothing
	Set objScan = Nothing
	Set objAppr = Nothing
End Sub

' Function to save data to CSV
Sub SaveToCSV(filename, refData, scanData)
    Dim objFS, objFile
    Set objFS = CreateObject("Scripting.FileSystemObject")
    Set objFile = objFS.CreateTextFile(filename, True)
    objFile.Write refData
    objFile.Write scanData
    objFile.Close
    Set objFile = Nothing
    Set objFS = Nothing
End Sub