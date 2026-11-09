(function(global){
  "use strict";

  function contentText(content){
    if(typeof content === "string") return content;
    if(!Array.isArray(content)) return "";
    var parts = [];
    for(var i=0;i<content.length;i++){
      var part = content[i];
      if(part && typeof part.text === "string") parts.push(part.text);
    }
    return parts.join("\n");
  }

  function mergeContents(a,b){
    if(typeof a === "string" && typeof b === "string") return a + "\n\n" + b;
    function toList(v){
      if(typeof v === "string") return v ? [{type:"text", text:v}] : [];
      if(Array.isArray(v)) return v.slice();
      return [];
    }
    return toList(a).concat(toList(b));
  }

  function normalizeHistory(messages){
    if(!Array.isArray(messages)) return [];
    var out = [];
    for(var i=0;i<messages.length;i++){
      var msg = messages[i];
      if(!msg || typeof msg.role !== "string") continue;
      var role = msg.role;
      var content = msg.content;
      // Drop empty assistant
      if(role === "assistant"){
        var txt = contentText(content);
        var hasImage = Array.isArray(content) && content.some(function(p){ return p && (p.type==="image_url"||p.type==="input_image"||p.image_url); });
        if(!txt.trim() && !hasImage){
          continue;
        }
      }
      if(out.length && out[out.length-1].role === role){
        // merge consecutive same-role
        var prev = out[out.length-1];
        prev.content = mergeContents(prev.content, content);
      } else {
        out.push({role: role, content: content});
      }
    }
    return out;
  }

  function applyEvent(messageState, event){
    // messageState: {messages: Array, streaming: object?}
    // event: payload from DB: {type, step, content, reasoning_content, reasoning_type, draft, kind, etc, event_run, event_seq, id}
    // Returns new state idempotent per event id
    if(!messageState) messageState = {messages: [], seenIds: {}, draft: "", reasoning: "", thinking: ""};
    if(!messageState.seenIds) messageState.seenIds = {};
    var id = event.id || event.event_run + ":" + event.event_seq;
    if(id && messageState.seenIds[id]) return messageState;
    if(id) messageState.seenIds[id]=true;
    var type = event.type;
    if(type === "delta"){
      var draft = event.draft === true;
      if(event.content != null){
        var txt = String(event.content);
        if(draft){
          // provisional
          messageState.draft = (messageState.draft || "") + txt;
        } else {
          // committed (if already has draft moved?)
          // For backward compat, treat as committed
          var last = messageState.messages[messageState.messages.length-1];
          if(last && last.role==="assistant" && !last.committed){
            // ensure assistant message exists
          }
          // Append to last assistant message or create
          if(!messageState.messages.length || messageState.messages[messageState.messages.length-1].role !== "assistant"){
            messageState.messages.push({role:"assistant", content:"", uid: "msg-"+Date.now()});
          }
          var cur = messageState.messages[messageState.messages.length-1];
          cur.content = (cur.content || "") + txt;
        }
      }
      if(event.reasoning_content != null){
        var rc = String(event.reasoning_content);
        // reasoning goes to thinking/COT
        if(!messageState.messages.length || messageState.messages[messageState.messages.length-1].role !== "assistant"){
          messageState.messages.push({role:"assistant", content:"", thinking:"", uid: "msg-"+Date.now()});
        }
        var lastAssistant = messageState.messages[messageState.messages.length-1];
        lastAssistant.thinking = (lastAssistant.thinking || "") + rc;
        lastAssistant.thinkingType = event.reasoning_type || lastAssistant.thinkingType || "raw";
      }
    } else if(type === "step_commit"){
      var kind = event.kind;
      if(kind === "answer"){
        // move draft to committed content
        if(messageState.draft){
          if(!messageState.messages.length || messageState.messages[messageState.messages.length-1].role !== "assistant"){
            messageState.messages.push({role:"assistant", content:"", uid:"msg-"+Date.now()});
          }
          var last = messageState.messages[messageState.messages.length-1];
          last.content = (last.content || "") + messageState.draft;
          last.committed = true;
          messageState.draft = "";
        }
      } else if(kind === "reasoning"){
        // move draft into COT
        if(messageState.draft){
          if(!messageState.messages.length || messageState.messages[messageState.messages.length-1].role !== "assistant"){
            messageState.messages.push({role:"assistant", content:"", thinking:"", uid:"msg-"+Date.now()});
          }
          var la = messageState.messages[messageState.messages.length-1];
          la.thinking = (la.thinking || "") + messageState.draft;
          la.thinkingType = "mixed";
          messageState.draft = "";
        }
      }
    } else if(type === "step_reset"){
      // discard current drafts of that step
      messageState.draft = "";
    } else if(type === "error"){
      // preserve existing content, append inline error note
      if(messageState.messages.length){
        var em = messageState.messages[messageState.messages.length-1];
        em.error = event.message || event.code || "Hiba";
        // Do not delete content
      } else {
        messageState.messages.push({role:"assistant", content:"", error: event.message});
      }
    } else if(type === "terminal" || type === "done"){
      // Mark status
      messageState.terminal = event.outcome || event.status || "completed";
      messageState.done = true;
    } else if(type === "progress"){
      messageState.progress = event.message;
    } else if(type === "agent_event"){
      // ignore for message state but keep
      if(!messageState.agentEvents) messageState.agentEvents=[];
      messageState.agentEvents.push(event);
    }
    return messageState;
  }

  function branchHelpers(){
    // placeholder for branch logic - not needed for tests
    return {
      fork: function(messages, index){
        return messages.slice(0,index);
      }
    };
  }

  var exported = {
    normalizeHistory: normalizeHistory,
    applyEvent: applyEvent,
    branchHelpers: branchHelpers,
    contentText: contentText
  };

  if(typeof module !== "undefined" && module.exports){
    module.exports = exported;
  } else {
    global.ChatCore = exported;
    global.normalizeHistory = normalizeHistory;
    global.applyEvent = applyEvent;
  }
})(typeof window !== "undefined" ? window : typeof global !== "undefined" ? global : this);
