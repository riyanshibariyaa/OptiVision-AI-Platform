var Common =
{
	LoadView: function()
	{
		var GivenView = window.location.hash.slice(1);
		if (GivenView == "")
		{
			window.location = "/home#dashboard";
		}
		else
		{
			$(".main-container").load("/home/" + GivenView ,
				function ()
				{
					CurrentView.Init();
				}
			);
		}
	},

	Debug: function(GivenString)
	{
		if (GlbDebug)
		{
			console.log(new Date().getHours() + ":" + new Date().getMinutes() + ":" + new Date().getSeconds() + ":" + new Date().getMilliseconds() + ": " + GivenString);
		}
	},

	showToast: function(message, type) 
	{ 
		// Define message type classes for Bootstrap toasts
		const typeClasses = {
			error: 'text-bg-danger',
			info: 'text-bg-info',
			warning: 'text-bg-warning'
		};
	
		// Create toast container if it doesn't already exist
		if ($('.toast-container').length === 0) {
			$('body').append('<div class="toast-container position-fixed bottom-0 end-0 p-3"></div>');
		}
	
		// Dynamically generate the toast HTML
		const toastHtml = `
			<div class="toast align-items-center ${typeClasses[type] || 'text-bg-info'} border-0" role="alert" aria-live="assertive" aria-atomic="true" data-bs-delay="3000">
				<div class="d-flex">
					<div class="toast-body">
						${message}
					</div>
					<button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
				</div>
			</div>
		`;
	
		// Append the toast HTML to the container
		const $toastElement = $(toastHtml).appendTo('.toast-container');
	
		// Initialize and show the toast
		const toast = new bootstrap.Toast($toastElement[0]);
		toast.show();
	
		// Automatically remove the toast from the DOM when it hides
		$toastElement.on('hidden.bs.toast', function () {
			$(this).remove();
		});
	},
	
	PrvntEvtPropg: function ()
	{
		//This function ensures that default propagation of click is stopped with the handling of this event
		event.preventDefault();
		return false;
	},

	UpdateUnits: function ()
    {
        Requester.Send("get_units", {"GlbSelectedUnitId":GlbSelectedUnitId}, function(Error, Data)
        {
            if (Error)
            {
                location.href = "/?RCode=3";
            }   
            else
            {
                RetVal = Data;
                
                Common.loadTemplate('/static/handlebarTemplates/Units.hbs', RetVal, "#UniteSelection");
                GlbSelectedUnitId = RetVal[0]["_id"];
                GlbSelectedUnitCode = RetVal[0]["name"];
        
                GlbUnits = RetVal.map(function (data) { return data["_id"] });
        
                $("#UniteSelection").on("change", function () 
                {
                    GlbSelectedUnitId = $(this).val();
                    GlbSelectedUnitCode = $(this).find(':selected').data("code");
                    Common.UpdateMenu(InitializeMenu = true);
                })
                Common.UpdateMenu(InitializeMenu = true);
            }
        });

        
    },

    UpdateMenu: function (InitializeMenu = false)
    {
        Requester.Send("get_menus", {"GlbSelectedUnitId":GlbSelectedUnitId}, function(Error, Data)
        {
            if (Error)
            {
                location.href = "/?RCode=3";
            }   
            else
            {
                var RetVal = Data;
                var TempMenus = [];
                var MenuItem = null;
                var SubMenu = null;
                var GroupOrder = null;
                
                for (var key in RetVal) {
                    var Row = RetVal[key];
                    
                    // If the current row is a group (parent menu item)
                    if (Row.parent === null && Row.is_group) 
                    {
                        if (MenuItem !== null) 
                        {
                            TempMenus.push(MenuItem); // Push the previous group
                        }
                
                        // Initialize a new menu group
                        MenuItem = {};
                        MenuItem["MenuId"] = Row["_id"];
                        MenuItem["IconClass"] = Row.icon;
                        MenuItem["Title"] = Row.title;
                        MenuItem["Link"] = Row.view;
                        MenuItem["IsOpen"] = false;
                        MenuItem["HasSubMenu"] = false;
                        MenuItem["SubMenu"] = [];
                    } 
                    // Handle menu items with no submenus (is_group is false, parent is null)
                    else if (Row.parent === null && !Row.is_group) 
                    {
                        // If there's an open group item, push it first
                        if (MenuItem !== null) {
                            TempMenus.push(MenuItem);
                        }

                        // Add the standalone menu item
                        MenuItem = {};
                        MenuItem["MenuId"] = Row["_id"];
                        MenuItem["IconClass"] = Row.icon;
                        MenuItem["Title"] = Row.title;
                        MenuItem["Link"] = Row.view;
                        MenuItem["IsOpen"] = false;  // Standalone menus don't have submenus, so it's not 'open'
                        MenuItem["HasSubMenu"] = false; // No submenus for this menu
                        MenuItem["SubMenu"] = [];

                        TempMenus.push(MenuItem); // Push the standalone item directly
                        MenuItem = null; // Reset MenuItem
                    }
                    // If the current row is not a group and belongs to a parent group
                    else if (Row.parent !== null) 
                    {
                        MenuItem["HasSubMenu"] = true;
                        SubMenu = {};
                        SubMenu["MenuId"] = Row["_id"];
                        SubMenu["IconClass"] = Row.icon;
                        SubMenu["Title"] = Row.title;
                        SubMenu["Link"] = Row.view;

                        MenuItem["SubMenu"].push(SubMenu);
                    }
                }
                
                // Push the last MenuItem after looping
                if (MenuItem !== null) {
                    TempMenus.push(MenuItem);
                }
                
                // Set the first menu item as open by default
                if (TempMenus.length > 0) {
                    TempMenus[0]["IsOpen"] = true;
                    GlbSelectedMenuId = TempMenus[0]["MenuId"];
                    if (InitializeMenu)
                        window.location.hash = TempMenus[0]["Link"];
                }
                
                // Prepare the Menu object
                var Menu = {};
                Menu["Menus"] = TempMenus;
                
                
                // Compile the Handlebars template and apply it
                Common.loadTemplate('/static/handlebarTemplates/Menus.hbs', Menu, "#page-sidebar-menu");
                // var TmplMenuList = Handlebars.compile($("#TmplMenuList").html());
                // $("#page-sidebar-menu").html(TmplMenuList(Menu));
                
                // Handle active menu state
                $("a").click(function () {
                    GlbSelectedMenuId = $(this).attr("id");
                });
                
                $(".nav-item").click(function ()
                {
                    var ButtonClicked = $(this);
                    if (!ButtonClicked.hasClass("parent-menu")) {
                        $(".nav-item").removeClass("active ui-state-active ui-state-hover");
                        ButtonClicked.addClass("ui-state-active active");
                        ButtonClicked.parent().parent().addClass("ui-state-hover active");
                    } else if (!ButtonClicked.hasClass("HasSubMenu")) {
                        $(".nav-item").removeClass("active ui-state-active ui-state-hover");
                        ButtonClicked.addClass("ui-state-hover active");
                    } else if (ButtonClicked.hasClass("parent-menu") && ButtonClicked.hasClass("HasSubMenu") && !ButtonClicked.hasClass("active")) {
                        $(".nav-item .collapse").collapse("hide");
                        if (ButtonClicked.children('a').children('.Arrow').hasClass('fas fa-angle-left'))
                            ButtonClicked.children('a').children('.Arrow').attr('class', 'Arrow fas fa-angle-down pull-right');
                        else
                            ButtonClicked.children('a').children('.Arrow').attr('class', 'Arrow fas fa-angle-left pull-right');
                    }
                });
            } 
        });    
    },

    BindHashChange: function (CallBackFunction)
    {
        // 
        $(window).on("hashchange", CallBackFunction);
    },

    Confirm: function (GivenText, CallBack)
    {
        // Create modal elements
        var modal = document.createElement('div');
        modal.classList.add('modal', 'fade');
        modal.setAttribute('tabindex', '-1');
        modal.setAttribute('aria-labelledby', 'confirmModalLabel');
        modal.setAttribute('aria-hidden', 'true');

        modal.innerHTML = `
            <div class="modal-dialog">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title" id="confirmModalLabel">Confirm</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                    </div>
                    <div class="modal-body">
                        <p>${GivenText}</p>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">No</button>
                        <button type="button" class="btn btn-primary" id="yesBtn">Yes</button>
                    </div>
                </div>
            </div>
        `;

        document.body.appendChild(modal);

        // Initialize modal
        var modalInstance = new bootstrap.Modal(modal);
        modalInstance.show();

        // Add event listeners for the buttons
        document.getElementById('yesBtn').addEventListener('click', function() 
        {
            if (CallBack) 
            {
                CallBack('Yes');
            }
            modalInstance.hide();
            document.body.removeChild(modal);
        });

        if (CallBack) 
        {
            modal.querySelector('[data-bs-dismiss="modal"]').addEventListener('click', function() 
            {
                CallBack('No');
                modalInstance.hide();
                document.body.removeChild(modal);
            });
        } 
        else 
        {
            modal.querySelector('[data-bs-dismiss="modal"]').addEventListener('click', function() 
            {
                modalInstance.hide();
                document.body.removeChild(modal);
            });
        }
    },

    loadTemplate: function(templateUrl, data, target) 
    {
        $.get(templateUrl, function(templateSource) 
        {
            
            var template = Handlebars.compile(templateSource);
            var html = template(data);
            $(target).html(html);
        });
    }
};

// Helper functions and extensions
Handlebars.registerHelper('ifCond', function (v1, v2, options)
{
	if (v1 === v2)
	{
		return options.fn(this);
	}
	return options.inverse(this);
});

Handlebars.registerHelper('TrimWhitespace', function (aString) 
{
	return aString.replace(/\s+/g,''); 
});


function ClsPaginator()
{
    var This = this;
    this.TotalRecords = 0;
    this.PageSize = 50;
    this.PageLast = 0;
    this.PageCurrent = 0;
    this.PageOffset = 0;
    this.ButtonPagination;
    this.TextBoxPage;
    this.CallBackFunction;
    this.ActiveTimer;

    //Paginator.Init(".ButtonPagination", ".PanelPagination", ".ButtonFirst", ".ButtonLast", ".ButtonPrev", ".ButtonNext")
    this.Init = function (ButtonPagination, PanelPagination, ButtonFirst, ButtonLast, ButtonPrev, ButtonNext, TextBoxPage, CallBackFunction)
    {
        This.ButtonPagination = ButtonPagination;
        This.TextBoxPage = TextBoxPage;
        This.CallBackFunction = CallBackFunction;

        //$(ButtonPagination).click(function ()
        //{
        //	var ButtonPagination = $(this);
        //	$(PanelPagination).css(
        //	{
        //		top: ButtonPagination.offset().top + ButtonPagination.outerHeight() + 10,
        //		left: ButtonPagination.offset().left + ButtonPagination.width() - 209
        //	})
        //	$(PanelPagination).toggle();
        //});

        $(ButtonFirst).click(function ()
        {
            if (This.PageCurrent != 1)
            {
                This.PageCurrent = 1;
                This.PageOffset = (This.PageCurrent - 1) * This.PageSize;
                $(This.TextBoxPage).html(This.PageCurrent);
                This.Hide();
            }
        });

        $(ButtonLast).click(function ()
        {
            if (This.PageCurrent != This.PageLast)
            {
                This.PageCurrent = This.PageLast;
                This.PageOffset = (This.PageCurrent - 1) * This.PageSize;
                $(This.TextBoxPage).html(This.PageCurrent);
                This.Hide();
            }
        });

        $(ButtonPrev).click(function ()
        {
            if (This.PageCurrent != 1)
            {
                This.PageCurrent--;
                This.PageOffset = (This.PageCurrent - 1) * This.PageSize;
                $(This.TextBoxPage).html(This.PageCurrent);
                This.Hide();
            }
        });

        $(ButtonNext).click(function ()
        {
            if (This.PageCurrent != This.PageLast)
            {
                This.PageCurrent++;
                This.PageOffset = (This.PageCurrent - 1) * This.PageSize;
                $(This.TextBoxPage).html(This.PageCurrent);
                This.Hide();
            }
        });
    };

    this.Hide = function (ButtonClicked)
    {
        // Rest the timer so that any previous click is not called back on
        window.clearTimeout(This.ActiveTimer);

        // Call Callback only  if timeout has elapsed
        This.ActiveTimer = window.setTimeout(function ()
        {
            if (This.CallBackFunction)
            {
                This.CallBackFunction(ButtonClicked);
            }
        }, 400);
    };

    this.SetTotalRecords = function (GivenValue)
    {
        This.TotalRecords = GivenValue;
        This.PageLast = Math.ceil(This.TotalRecords / This.PageSize);
        This.PageCurrent = 1;
        This.PageOffset = 0;
        $(This.TextBoxPage).html(This.PageCurrent);
    }

    this.SetPageSize = function (GivenValue)
    {
        This.PageSize = GivenValue;
        This.PageLast = Math.ceil(This.TotalRecords / This.PageSize);
        This.PageCurrent = 1;
        This.PageOffset = 0;
        $(This.TextBoxPage).html(This.PageCurrent);
    }
}