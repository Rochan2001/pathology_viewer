var data = JSON.parse("{{uploaded_file_url|escapejs}}");
if(data){

    var viewer1 = OpenSeadragon({
      id: "openseadragon1",
      prefixUrl: "../openseadragon/images/",
      tileSources: "../mydz.dzi",
      visibilityRatio: 1.0,
      constrainDuringPan: true,
      debugMode: false,
    });

}
