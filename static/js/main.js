
function loadFile(input) {

//    var reader = new FileReader();
//    reader.onload = function (e) {
//        $('.img-upload')
//            .attr('src', e.target.result);
//    };
//
//    reader.readAsDataURL(input.target.files[0]);

    var frm = new FormData();
    frm.append('file', input.target.files[0]);

    $.ajax({
        url: '/upload',
        type: 'POST',
        data: frm,
        async: true,
        contentType: false,
        processData: false,
        success: function (data) {
            var url = "/?filename=" + data.filename;
            console.log(url);
            window.location.href = url;
        },
        error: function (data) {
            console.log(data);
        },
        cache: false
    });
};
