(window["webpackJsonp"]=window["webpackJsonp"]||[]).push([["systemConfig"],{"3cfd":function(e,t,r){},"3e3d":function(e,t,r){"use strict";r.r(t);var s=function(){var e=this,t=e.$createElement,r=e._self._c||t;return r("section",{staticClass:"container-wrapper"},[r("div",{staticClass:"center-form system-config-form"},[r("div",{staticClass:"form-group"},[r("header",{staticClass:"group-header"},[e._v("授权状态")]),r("div",{staticClass:"group-body"},[r("div",{staticClass:"group-item"},[r("label",{staticClass:"item-title"},[e._v("授权信息：")]),e.licenseInfo.isAuthorized?r("span",{staticClass:"item-con success"},[e._v("已授权")]):r("span",{staticClass:"item-con fail"},[e._v("未授权")])]),r("div",{staticClass:"group-item"},[r("label",{staticClass:"item-title"},[e._v("授权方式：")]),r("span",{staticClass:"item-con"},[r("el-select",{attrs:{placeholder:"请选择授权方式"},model:{value:e.licenseType,callback:function(t){e.licenseType=t},expression:"licenseType"}},e._l(e.licenseTypeList,(function(e){return r("el-option",{key:e.value,attrs:{label:e.name,value:e.value}})})),1)],1)]),"licenseCode"===e.licenseType?[r("div",{staticClass:"group-item"},[r("label",{staticClass:"item-title"},[e._v("授权申请码：")]),r("span",{staticClass:"item-con",attrs:{title:"点击复制授权申请码"},on:{click:function(t){return t.stopPropagation(),e.copyFn(e.licenseInfo.requestCode)}}},[e._v(e._s(e.licenseInfo.requestCode))])]),r("div",{staticClass:"group-item"},[r("label",{staticClass:"item-title"}),r("div",{staticClass:"item-con w330"},[r("el-input",{staticClass:"input-area",attrs:{type:"textarea",placeholder:"请输入授权码",clearable:"",autosize:{minRows:3,maxRows:10}},model:{value:e.licenseCode,callback:function(t){e.licenseCode="string"===typeof t?t.trim():t},expression:"licenseCode"}})],1)])]:"serverUrl"===e.licenseType?[r("div",{staticClass:"group-item"},[r("label",{staticClass:"item-title "},[e._v("服务器URL：")]),r("span",{staticClass:"item-con",class:{"is-error":!e.isValidServerUrl}},[r("el-input",{attrs:{placeholder:"请输入合法的服务器URL"},on:{blur:function(t){return e.validateServerUrl(e.serverUrl)}},model:{value:e.serverUrl,callback:function(t){e.serverUrl="string"===typeof t?t.trim():t},expression:"serverUrl"}}),r("span",{directives:[{name:"show",rawName:"v-show",value:!e.isValidServerUrl,expression:"!isValidServerUrl"}],staticClass:"error-tip"},[e._v("格式不合法")])],1)])]:e._e(),""!==e.licenseType?r("div",{staticClass:"group-item"},[r("label",{staticClass:"item-title"}),r("el-button",{staticClass:"mt10",attrs:{type:"primary"},on:{click:e.saveLicenseStatus}},[e._v("保存")])],1):e._e()],2)])])])},n=[],i=(r("8e6e"),r("ac6a"),r("456d"),r("96cf"),r("1da1")),a=r("ade3"),c=r("2f62");function o(e,t){var r=Object.keys(e);if(Object.getOwnPropertySymbols){var s=Object.getOwnPropertySymbols(e);t&&(s=s.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),r.push.apply(r,s)}return r}function l(e){for(var t=1;t<arguments.length;t++){var r=null!=arguments[t]?arguments[t]:{};t%2?o(Object(r),!0).forEach((function(t){Object(a["a"])(e,t,r[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(r)):o(Object(r)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(r,t))}))}return e}var u=/((([A-Za-z]{3,9}:(?:\/\/)?)(?:[\-;:&=\+\$,\w]+@)?[A-Za-z0-9\.\-]+|(?:www\.|[\-;:&=\+\$,\w]+@)[A-Za-z0-9\.\-]+)((?:\/[\+~%\/\.\w\-_]*)?\??(?:[\-\+=&;%@\.\w_]*)#?(?:[\.\!\/\\\w]*))?)/,p={name:"SystemConfig",data:function(){return{licenseType:"licenseCode",licenseTypeList:[{name:"激活码",value:"licenseCode"},{name:"认证服务器",value:"serverUrl"}],licenseCode:"",serverUrl:"",isValidServerUrl:!0}},computed:{licenseInfo:function(){return this.$store.state.licenseInfo}},created:function(){this.getMpgsUrl()},methods:l(l({},Object(c["b"])(["httpGet","httpPost"])),{},{getMpgsUrl:function(){var e=Object(i["a"])(regeneratorRuntime.mark((function e(){var t;return regeneratorRuntime.wrap((function(e){while(1)switch(e.prev=e.next){case 0:return e.next=2,this.httpGet(["/algorithm_server/mpgs_url"]);case 2:t=e.sent,t&&(this.serverUrl=t);case 4:case"end":return e.stop()}}),e,this)})));function t(){return e.apply(this,arguments)}return t}(),saveLicenseStatus:function(){var e=this.licenseType;"licenseCode"===e?this.submitLicenseFn():"serverUrl"===e&&this.submitMpgsUrlFn()},submitLicenseFn:function(){var e=Object(i["a"])(regeneratorRuntime.mark((function e(){var t,r;return regeneratorRuntime.wrap((function(e){while(1)switch(e.prev=e.next){case 0:if(t=this.licenseCode,""!==t){e.next=3;break}return e.abrupt("return");case 3:return r={license:t},e.prev=4,e.next=7,this.httpPost(["/algorithm_server/license",r]);case 7:e.sent,this.$store.dispatch("getLicenseInfo",!0),window._.showToast("提交成功！"),e.next=15;break;case 12:e.prev=12,e.t0=e["catch"](4),window._.showToast("提交失败！","error");case 15:case"end":return e.stop()}}),e,this,[[4,12]])})));function t(){return e.apply(this,arguments)}return t}(),submitMpgsUrlFn:function(){var e=Object(i["a"])(regeneratorRuntime.mark((function e(){var t,r,s;return regeneratorRuntime.wrap((function(e){while(1)switch(e.prev=e.next){case 0:if(t=this.serverUrl,""!==t&&this.isValidServerUrl){e.next=3;break}return e.abrupt("return");case 3:return r={url:t},e.prev=4,e.next=7,this.httpPost(["/algorithm_server/mpgs_url",r,{notInData:!0}]);case 7:s=e.sent,console.log(s),this.$store.dispatch("getLicenseInfo",!0),window._.showToast("提交成功！"),e.next=16;break;case 13:e.prev=13,e.t0=e["catch"](4),window._.showToast("提交失败！","error");case 16:case"end":return e.stop()}}),e,this,[[4,13]])})));function t(){return e.apply(this,arguments)}return t}(),copyFn:function(){var e=Object(i["a"])(regeneratorRuntime.mark((function e(t){return regeneratorRuntime.wrap((function(e){while(1)switch(e.prev=e.next){case 0:return e.prev=0,e.next=3,navigator.clipboard.writeText(t);case 3:window._.showToast("已复制","success",600),e.next=9;break;case 6:e.prev=6,e.t0=e["catch"](0),window._.showToast("浏览器不支持该功能，请手动复制","error");case 9:case"end":return e.stop()}}),e,null,[[0,6]])})));function t(t){return e.apply(this,arguments)}return t}(),validateServerUrl:function(e){this.isValidServerUrl=u.test(e)}})},v=p,d=(r("459a"),r("2877")),f=Object(d["a"])(v,s,n,!1,null,"124174b6",null);t["default"]=f.exports},"459a":function(e,t,r){"use strict";r("3cfd")}}]);