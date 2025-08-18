class ClsRequestAdapter
{
	constructor(){}

    Send(Id, Data, CallBack)
    {
        $.ajax(
        {
            async: true,
            method: "POST",
            type: "POST",
            cache: false,
            dataType: "json",
            contentType: "application/json",
            headers:
            {
                'Authorization': 'Bearer ' + localStorage.getItem('token'),
                "Content-Type": "application/json"
            },

            url: "/" + Id,
            data: JSON.stringify(Data),
            success: function(Data) 
            {
                CallBack(false, Data);
            },
            error: function(Error) 
            {
                CallBack(true,Error.responseJSON.message);
            }
        });
    }
}

//Instantiate An Object for the class
const Requester = new ClsRequestAdapter();

