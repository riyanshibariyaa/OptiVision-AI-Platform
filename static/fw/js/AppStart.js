var GlbServer;
var GlbPopupIndex = 0;
var GlbDebug = true;
var GlbUsers = [];
var GlbUnits = [];
var GlbSelectedMenuId = "";
var GlbSelectedUnitId = "";
var GlbSelectedUnitCode = "";


// Application begins here
$(function ()
{
	Common.Debug("Page Load Complete");
	
	Common.UpdateUnits();
	Common.BindHashChange(Common.LoadView);

	Common.LoadView();
});
