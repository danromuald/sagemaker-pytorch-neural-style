//Copyright 2013-2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
//Licensed under the Apache License, Version 2.0 (the "License").
//You may not use this file except in compliance with the License.
//A copy of the License is located at
//
//    http://aws.amazon.com/apache2.0/
//
//or in the "license" file accompanying this file. This file is distributed
//on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
//either express or implied. See the License for the specific language
//governing permissions and limitations under the License.

// Main app config
var opticaiConfig = {
  apiURL: 'http://ec2-34-203-210-79.compute-1.amazonaws.com:8080/invocations',
  cdn: "http://d1x7gg3vffpqew.cloudfront.net/",
  errorContainer: $('#errorMessage'),
  loader: $('#loader'),
  debug: false,
  modals: {
    transferModal: $('#transferModal'),
    stylesModal: $('#stylesModal')
  },
  buttons: {
    uploadButton: $('#uploadButton'),
    styleButton: $('#styleButton'),
    transferButton: $('#transferButton'),
    downloadButton: $('#downloadButton')
  },
  inputs: {
    photoInput: $('#photoInput'),
    styleInput: $('#styleInput')
  },
  photos: {
    userPhoto: $('#userPhoto'),
    stylePhoto: $('#stylePhoto'),
    styledPhoto: $('#styledPhoto')
  },
  steps: {
    userPhotoStep: $('#userPhotoStep'),
    styleStep: $('#styleStep'),
    transferStep: $('#transferStep')
  }
}
// Main app controller
var opticaiController = {
  init: function() {
    if(opticaiConfig.debug) {
      console.time("opticaiController: App loaded");
    }
    opticaiController.main();
  },
  main: function() {
    opticaiController.log("opticaiController: main: start");
    opticaiController.initializeModals();
    opticaiController.log("opticaiController: main: complete");
    if(opticaiConfig.debug) {
      console.timeEnd("opticaiController: App loaded");
    }
  },
  initializeModals: function() {
    opticaiController.log("opticaiController: initializeModals: start");
    opticaiConfig.modals.transferModal.modal({
      onHide: function() {
        opticaiController.resetView();
      }
    });
    opticaiController.log("opticaiController: initializeModals: complete");
    opticaiController.initializeBindings();
  },
  initializeBindings: function() {
    opticaiController.log("opticaiController: initializeBindings: start");
    // Upload button
    opticaiConfig.buttons.uploadButton.on('click', function() {
      opticaiController.log("opticaiController: userPhotoStep: start");
      opticaiConfig.inputs.photoInput.click();
    });
    // Select style button
    opticaiConfig.buttons.styleButton.on('click', function() {
      opticaiController.log("opticaiController: styleStep: start");
      opticaiConfig.modals.stylesModal.modal('show');
    });
    // Select style photo
    $('.styleImages img').on('click', function(e) {
      var src = e.currentTarget.src;

      opticaiConfig.inputs.styleInput.val(src);
      opticaiConfig.photos.stylePhoto.attr('src', src);
      opticaiConfig.modals.stylesModal.modal('hide');

      // Enable the next step
      opticaiController.updateSteps({
        userPhotoStep: false,
        styleStep: false,
        transferStep: true
      });

      opticaiController.log("opticaiController: styleStep: complete");
    });
    // Transfer button
    opticaiConfig.buttons.transferButton.on('click', function(e) {
      // Reset loader and white-image
      $('#loader').addClass('active');
      opticaiConfig.photos.styledPhoto.attr('src', opticaiConfig.cdn + 'white-image.png');

      opticaiConfig.modals.transferModal.modal('show');
      opticaiController.transfer();
    });
    // Photo Input listener
    opticaiController.photoInputBinding();

    opticaiController.log("opticaiController: initializeBindings: complete");
  },
  setInputWithPhotoURLFromUpload: function(input, imageContainer) {
    if (input.files && input.files[0]) {
      var reader = new FileReader();
      var file = input.files[0];

      reader.onload = function(e) {
        imageContainer.attr('src', e.target.result);
        opticaiController.log("opticaiController: setInputWithPhotoURLFromUpload: complete");

        // Enable the next step
        opticaiController.updateSteps({
          userPhotoStep: false,
          styleStep: true,
          transferStep: false
        });
        opticaiController.log("opticaiController: userPhotoStep: complete");
      }
      reader.readAsDataURL(input.files[0]);
    }
  },
  photoInputBinding: function() {
    opticaiConfig.inputs.photoInput.change(function(data) {
      var input = data.currentTarget
      // Display the selected user photo
      opticaiController.setInputWithPhotoURLFromUpload(
        input,
        opticaiConfig.photos.userPhoto
      );
    });
  },
  transfer: function() {
    opticaiController.log("opticaiController: transfer: start");
    // Build JSON payload with:
    // - (String) styledPhotoURL
    // - (base64 BLOB) userPhoto
    var stylePhoto = opticaiConfig.inputs.styleInput.val().split("/")[opticaiConfig.inputs.styleInput.val().split("/").length - 1];
    var jsonPayload = {
      stylePhoto: stylePhoto,
      userPhoto: opticaiConfig.photos.userPhoto.attr('src')
    };
    // Send POST request to API endpoint
    $.ajax({
      url: opticaiConfig.apiURL,
      data: JSON.stringify(jsonPayload),
      type: 'POST',
      contentType: "application/json",
      success: function(data){
        opticaiConfig.loader.removeClass('active');
        opticaiConfig.photos.styledPhoto.attr('src', data['stylizedPhoto']);
        opticaiConfig.buttons.downloadButton.data('download', "Styled_photo_" + stylePhoto);
        opticaiConfig.buttons.downloadButton.attr('href', data['stylizedPhoto']);
      },
      error: function(error, message) {
        opticaiConfig.loader.removeClass('active');
        opticaiConfig.photos.styledPhoto.attr('src', opticaiConfig.cdn + "white-image.png");
        opticaiConfig.buttons.downloadButton.attr('href', "");
        opticaiConfig.errorContainer.show();
      }
    });
    opticaiController.log("opticaiController: transfer: complete");
  },
  updateSteps: function(steps) {
    // Default state should be {userPhotoStep: true, styleStep: false, transferStep: false}
    if(steps.userPhotoStep == true) {
      opticaiConfig.steps.userPhotoStep
        .removeClass('disabled')
        .addClass('active');

      opticaiConfig.buttons.uploadButton
        .removeClass('basic')
        .removeClass('disabled');
    } else {
      opticaiConfig.steps.userPhotoStep
        .removeClass('active')
        .addClass('disabled');

      opticaiConfig.buttons.uploadButton
        .removeClass('active')
        .addClass('basic')
        .addClass('disabled');
    }

    if(steps.styleStep == true) {
      opticaiConfig.steps.styleStep
        .removeClass('disabled')
        .addClass('active');

      opticaiConfig.buttons.styleButton
        .removeClass('basic')
        .removeClass('disabled');;
    } else {
      opticaiConfig.steps.styleStep
        .removeClass('active')
        .addClass('disabled');

      opticaiConfig.buttons.styleButton
        .removeClass('active')
        .addClass('basic')
        .addClass('disabled');
    }

    if(steps.transferStep == true) {
      opticaiConfig.steps.transferStep
        .removeClass('disabled')
        .addClass('active');

      opticaiConfig.buttons.transferButton
        .removeClass('basic')
        .removeClass('disabled');;
    } else {
      opticaiConfig.steps.transferStep
        .removeClass('active')
        .addClass('disabled');

      opticaiConfig.buttons.transferButton
        .removeClass('active')
        .addClass('basic')
        .addClass('disabled');
    }
  },
  resetView: function() {
    opticaiController.log("opticaiController: resetView: start");
    // Reset inputs
    opticaiConfig.inputs.styleInput.val("");
    opticaiConfig.inputs.photoInput.replaceWith(opticaiConfig.inputs.photoInput.val('').clone(true));
    opticaiController.photoInputBinding();
    // Reset photos
    opticaiConfig.photos.userPhoto.attr('src', opticaiConfig.cdn + 'white-image.png');
    opticaiConfig.photos.stylePhoto.attr('src', opticaiConfig.cdn + 'white-image.png');
    // Reset download button
    opticaiConfig.buttons.downloadButton.attr('href', "");
    // Reset error errorMessage
    opticaiConfig.errorContainer.hide();
    // Set the active step to the first one
    opticaiController.updateSteps({
      userPhotoStep: true,
      styleStep: false,
      transferStep: false
    });

    opticaiController.log("opticaiController: resetView: complete");
    return;
  },
  log: function(message) {
    if(opticaiConfig.debug) {
      console.info(message);
    }
    return;
  }
}
// Start app when DOM is ready.
$(document).ready( function() {
  opticaiController.init();
});
